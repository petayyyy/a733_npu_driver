#define _POSIX_C_SOURCE 200809L

#include <vip_lite.h>

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#ifdef A733_VIP_LEGACY_DEVICE_ID
#define A733_VIP_NETWORK_PROP_SET_DEVICE VIP_NETWORK_PROP_SET_DEVICE_ID
#else
#define A733_VIP_NETWORK_PROP_SET_DEVICE VIP_NETWORK_PROP_SET_DEVICE_INDEX
#endif

typedef struct {
    vip_network network;
    vip_buffer input;
    vip_buffer output;
    vip_buffer_create_params_t input_param;
    vip_buffer_create_params_t output_param;
    uint32_t input_elements;
    uint32_t output_elements;
    uint32_t input_bytes;
    uint32_t output_bytes;
    int prepared;
} stage_t;

static uint64_t now_us(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
}

static uint32_t element_count(const vip_buffer_create_params_t *param)
{
    uint32_t i;
    uint32_t count = 1;
    for (i = 0; i < param->num_of_dims; i++) {
        count *= param->sizes[i];
    }
    return count;
}

static void print_dims(const vip_buffer_create_params_t *param)
{
    uint32_t i;
    for (i = 0; i < param->num_of_dims; i++) {
        printf("%s%u", i == 0 ? "" : "x", param->sizes[i]);
    }
}

static void *read_file(const char *path, uint32_t *size_out)
{
    FILE *fp = fopen(path, "rb");
    long size;
    void *data;

    if (fp == NULL) {
        fprintf(stderr, "failed to open %s: %s\n", path, strerror(errno));
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    if (size <= 0) {
        fprintf(stderr, "invalid NBG size for %s: %ld\n", path, size);
        fclose(fp);
        return NULL;
    }
    fseek(fp, 0, SEEK_SET);

    data = malloc((size_t)size);
    if (data == NULL) {
        fprintf(stderr, "failed to allocate %ld bytes for %s\n", size, path);
        fclose(fp);
        return NULL;
    }
    if (fread(data, 1, (size_t)size, fp) != (size_t)size) {
        fprintf(stderr, "failed to read %s\n", path);
        free(data);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    *size_out = (uint32_t)size;
    return data;
}

static vip_status_e create_buffer_from_query(
    vip_network network,
    int is_input,
    vip_buffer_create_params_t *param,
    vip_buffer *buffer)
{
    vip_status_e status;

    memset(param, 0, sizeof(*param));
    param->memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT;
    if (is_input) {
        vip_query_input(network, 0, VIP_BUFFER_PROP_DATA_FORMAT, &param->data_format);
        vip_query_input(network, 0, VIP_BUFFER_PROP_NUM_OF_DIMENSION, &param->num_of_dims);
        vip_query_input(network, 0, VIP_BUFFER_PROP_SIZES_OF_DIMENSION, param->sizes);
        vip_query_input(network, 0, VIP_BUFFER_PROP_QUANT_FORMAT, &param->quant_format);
        vip_query_input(network, 0, VIP_BUFFER_PROP_FIXED_POINT_POS,
                        &param->quant_data.dfp.fixed_point_pos);
        vip_query_input(network, 0, VIP_BUFFER_PROP_TF_SCALE, &param->quant_data.affine.scale);
        vip_query_input(network, 0, VIP_BUFFER_PROP_TF_ZERO_POINT,
                        &param->quant_data.affine.zeroPoint);
    } else {
        vip_query_output(network, 0, VIP_BUFFER_PROP_DATA_FORMAT, &param->data_format);
        vip_query_output(network, 0, VIP_BUFFER_PROP_NUM_OF_DIMENSION, &param->num_of_dims);
        vip_query_output(network, 0, VIP_BUFFER_PROP_SIZES_OF_DIMENSION, param->sizes);
        vip_query_output(network, 0, VIP_BUFFER_PROP_QUANT_FORMAT, &param->quant_format);
        vip_query_output(network, 0, VIP_BUFFER_PROP_FIXED_POINT_POS,
                         &param->quant_data.dfp.fixed_point_pos);
        vip_query_output(network, 0, VIP_BUFFER_PROP_TF_SCALE, &param->quant_data.affine.scale);
        vip_query_output(network, 0, VIP_BUFFER_PROP_TF_ZERO_POINT,
                         &param->quant_data.affine.zeroPoint);
    }

    status = vip_create_buffer(param, sizeof(*param), buffer);
    return status;
}

static int stage_init(stage_t *s, const char *nbg_path, int stage_idx, uint32_t device_index)
{
    vip_status_e status;
    uint32_t nbg_size = 0;
    void *nbg_data;
    uint32_t input_count = 0;
    uint32_t output_count = 0;
    uint32_t mem_pool_size = 0;
    uint32_t core_count = 0;
    uint64_t t0;

    memset(s, 0, sizeof(*s));

    t0 = now_us();
    nbg_data = read_file(nbg_path, &nbg_size);
    if (nbg_data == NULL) {
        return -1;
    }
    printf("[stage %d] nbg_size=%u bytes\n", stage_idx, nbg_size);

    status = vip_create_network(nbg_data, nbg_size, VIP_CREATE_NETWORK_FROM_MEMORY, &s->network);
    if (status != VIP_SUCCESS) {
        fprintf(stderr, "[stage %d] vip_create_network failed: %d\n", stage_idx, status);
        free(nbg_data);
        return -1;
    }
    printf("[stage %d] create_network_us=%" PRIu64 "\n", stage_idx, now_us() - t0);

    {
        uint32_t dev_idx = device_index;
        status = vip_set_network(s->network, A733_VIP_NETWORK_PROP_SET_DEVICE, (void *)&dev_idx);
    }
    if (status != VIP_SUCCESS) {
        fprintf(stderr, "[stage %d] vip_set_network(device_index) failed: %d\n", stage_idx, status);
        return -1;
    }

    vip_query_network(s->network, VIP_NETWORK_PROP_INPUT_COUNT, &input_count);
    vip_query_network(s->network, VIP_NETWORK_PROP_OUTPUT_COUNT, &output_count);
    if (input_count != 1 || output_count != 1) {
        fprintf(stderr, "[stage %d] expected 1 input/1 output, got %u/%u\n",
                stage_idx, input_count, output_count);
        return -1;
    }

    if (create_buffer_from_query(s->network, 1, &s->input_param, &s->input) != VIP_SUCCESS) {
        fprintf(stderr, "[stage %d] create_buffer(input) failed\n", stage_idx);
        return -1;
    }
    if (create_buffer_from_query(s->network, 0, &s->output_param, &s->output) != VIP_SUCCESS) {
        fprintf(stderr, "[stage %d] create_buffer(output) failed\n", stage_idx);
        return -1;
    }

    s->input_elements = element_count(&s->input_param);
    s->output_elements = element_count(&s->output_param);
    s->input_bytes = vip_get_buffer_size(s->input);
    s->output_bytes = vip_get_buffer_size(s->output);

    printf("[stage %d] input_dims=", stage_idx);
    print_dims(&s->input_param);
    printf(" input_format=%d input_quant=%d input_elements=%u input_bytes=%u\n",
           s->input_param.data_format, s->input_param.quant_format,
           s->input_elements, s->input_bytes);
    printf("[stage %d] output_dims=", stage_idx);
    print_dims(&s->output_param);
    printf(" output_format=%d output_quant=%d output_dfp=%d output_elements=%u output_bytes=%u\n",
           s->output_param.data_format, s->output_param.quant_format,
           s->output_param.quant_data.dfp.fixed_point_pos, s->output_elements, s->output_bytes);

    if (s->output_bytes == 0 || s->input_bytes == 0) {
        fprintf(stderr, "[stage %d] zero buffer size\n", stage_idx);
        return -1;
    }

    vip_query_network(s->network, VIP_NETWORK_PROP_MEMORY_POOL_SIZE, &mem_pool_size);
    vip_query_network(s->network, VIP_NETWORK_PROP_CORE_COUNT, &core_count);
    printf("[stage %d] memory_pool_bytes=%u network_core_count=%u\n",
           stage_idx, mem_pool_size, (unsigned)core_count);

    t0 = now_us();
    status = vip_prepare_network(s->network);
    if (status != VIP_SUCCESS) {
        fprintf(stderr, "[stage %d] vip_prepare_network failed: %d\n", stage_idx, status);
        return -1;
    }
    s->prepared = 1;
    printf("[stage %d] prepare_network_us=%" PRIu64 "\n", stage_idx, now_us() - t0);

    status = vip_set_input(s->network, 0, s->input);
    if (status != VIP_SUCCESS) {
        fprintf(stderr, "[stage %d] vip_set_input failed: %d\n", stage_idx, status);
        return -1;
    }
    status = vip_set_output(s->network, 0, s->output);
    if (status != VIP_SUCCESS) {
        fprintf(stderr, "[stage %d] vip_set_output failed: %d\n", stage_idx, status);
        return -1;
    }

    return 0;
}

static int stage_copy_output_to_input(stage_t *src, stage_t *dst)
{
    uint8_t *src_mapped;
    uint8_t *dst_mapped;

    if (src->output_bytes != dst->input_bytes) {
        fprintf(stderr, "buffer size mismatch: src_out=%u dst_in=%u\n",
                src->output_bytes, dst->input_bytes);
        return -1;
    }

    if (vip_flush_buffer(src->output, VIP_BUFFER_OPER_TYPE_INVALIDATE) != VIP_SUCCESS) {
        fprintf(stderr, "vip_flush_buffer(src->output) failed\n");
        return -1;
    }

    src_mapped = (uint8_t *)vip_map_buffer(src->output);
    if (src_mapped == NULL) {
        fprintf(stderr, "vip_map_buffer(src->output) failed\n");
        return -1;
    }

    dst_mapped = (uint8_t *)vip_map_buffer(dst->input);
    if (dst_mapped == NULL) {
        fprintf(stderr, "vip_map_buffer(dst->input) failed\n");
        vip_unmap_buffer(src->output);
        return -1;
    }

    memcpy(dst_mapped, src_mapped, src->output_bytes);

    vip_unmap_buffer(src->output);
    vip_unmap_buffer(dst->input);

    if (vip_flush_buffer(dst->input, VIP_BUFFER_OPER_TYPE_FLUSH) != VIP_SUCCESS) {
        fprintf(stderr, "vip_flush_buffer(dst->input) failed\n");
        return -1;
    }

    return 0;
}

static int write_input_from_file(stage_t *s, const char *input_path)
{
    FILE *fp = fopen(input_path, "rb");
    uint8_t *mapped;
    size_t read_size;

    if (fp == NULL) {
        fprintf(stderr, "failed to open input file %s: %s\n", input_path, strerror(errno));
        return -1;
    }

    mapped = (uint8_t *)vip_map_buffer(s->input);
    if (mapped == NULL) {
        fprintf(stderr, "vip_map_buffer(input) failed\n");
        fclose(fp);
        return -1;
    }

    read_size = fread(mapped, 1, s->input_bytes, fp);
    fclose(fp);

    if (read_size != s->input_bytes) {
        fprintf(stderr, "read %zu bytes from %s, expected %u\n",
                read_size, input_path, s->input_bytes);
        vip_unmap_buffer(s->input);
        return -1;
    }

    vip_unmap_buffer(s->input);

    if (vip_flush_buffer(s->input, VIP_BUFFER_OPER_TYPE_FLUSH) != VIP_SUCCESS) {
        fprintf(stderr, "vip_flush_buffer(input) failed\n");
        return -1;
    }

    return 0;
}

static int stage_run(stage_t *s, int stage_idx, vip_inference_profile_t *profile)
{
    vip_status_e status;
    uint64_t wall_us;
    uint64_t t0 = now_us();

    status = vip_run_network(s->network);
    wall_us = now_us() - t0;

    if (status != VIP_SUCCESS) {
        fprintf(stderr, "[stage %d] vip_run_network failed: %d\n", stage_idx, status);
        return -1;
    }

    if (profile != NULL) {
        vip_query_network(s->network, VIP_NETWORK_PROP_PROFILING, (void *)profile);
        printf("[stage %d] run: wall=%" PRIu64 "us profile=%uus cycle=%u\n",
               stage_idx, wall_us, profile->inference_time, profile->total_cycle);
    }

    return 0;
}

static float check_output_consistency(stage_t *s, int stage_idx, float *ref, uint32_t count)
{
    uint8_t *mapped;
    uint32_t i;
    float sum_abs_diff = 0.0f;
    float max_abs_diff = 0.0f;
    int16_t *values16;
    int fl;

    if (vip_flush_buffer(s->output, VIP_BUFFER_OPER_TYPE_INVALIDATE) != VIP_SUCCESS) {
        fprintf(stderr, "vip_flush_buffer(output) failed\n");
        return -1.0f;
    }

    mapped = (uint8_t *)vip_map_buffer(s->output);
    if (mapped == NULL) {
        return -1.0f;
    }

    fl = s->output_param.quant_data.dfp.fixed_point_pos;
    values16 = (int16_t *)mapped;
    for (i = 0; i < count && i < s->output_elements; i++) {
        float val = (float)values16[i] / (float)(1ULL << fl);
        float diff = val - ref[i];
        float abs_diff = diff > 0.0f ? diff : -diff;
        sum_abs_diff += abs_diff;
        if (abs_diff > max_abs_diff) max_abs_diff = abs_diff;
    }

    if (i > 0) {
        printf("[stage %d] vs reference: mean_abs_diff=%.6f max_abs_diff=%.6f (over %u elems)\n",
               stage_idx, sum_abs_diff / (float)i, max_abs_diff, i);
    }

    vip_unmap_buffer(s->output);
    return sum_abs_diff;
}

static void stage_destroy(stage_t *s)
{
    if (s->prepared && s->network != NULL) {
        vip_finish_network(s->network);
    }
    if (s->input != NULL) {
        vip_destroy_buffer(s->input);
    }
    if (s->output != NULL) {
        vip_destroy_buffer(s->output);
    }
    if (s->network != NULL) {
        vip_destroy_network(s->network);
    }
    memset(s, 0, sizeof(*s));
}

int main(int argc, char **argv)
{
    const char *block0_nbg = NULL;
    const char *block1_nbg = NULL;
    const char *block0_input = NULL;
    uint32_t device_index = 0;
    int iterations = 5;
    vip_status_e status;
    stage_t stage0, stage1;
    int i;

    for (int a = 1; a < argc; a++) {
        if (strcmp(argv[a], "--block0") == 0 && a + 1 < argc) block0_nbg = argv[++a];
        else if (strcmp(argv[a], "--block1") == 0 && a + 1 < argc) block1_nbg = argv[++a];
        else if (strcmp(argv[a], "--input") == 0 && a + 1 < argc) block0_input = argv[++a];
        else if (strcmp(argv[a], "--device") == 0 && a + 1 < argc) device_index = (uint32_t)atoi(argv[++a]);
        else if (strcmp(argv[a], "--iters") == 0 && a + 1 < argc) iterations = atoi(argv[++a]);
        else {
            fprintf(stderr, "Usage: %s --block0 NBG --block1 NBG --input FILE [--device N] [--iters N]\n", argv[0]);
            return 1;
        }
    }

    if (block0_nbg == NULL || block1_nbg == NULL || block0_input == NULL) {
        fprintf(stderr, "required: --block0, --block1, --input\n");
        return 1;
    }

    /* Initialize VIPLite */
    status = vip_init();
    if (status != VIP_SUCCESS) {
        fprintf(stderr, "vip_init failed: %d\n", status);
        return 1;
    }
    printf("vip_init=OK\n");

    /* Initialize both stages */
    if (stage_init(&stage0, block0_nbg, 0, device_index) != 0) {
        vip_destroy();
        return 1;
    }
    if (stage_init(&stage1, block1_nbg, 1, device_index) != 0) {
        stage_destroy(&stage0);
        vip_destroy();
        return 1;
    }

    /* Verify buffer sizes match for chaining */
    printf("\nChain check: stage0 output=%u bytes, stage1 input=%u bytes\n",
           stage0.output_bytes, stage1.input_bytes);
    if (stage0.output_bytes != stage1.input_bytes) {
        fprintf(stderr, "ERROR: buffer size mismatch for chaining\n");
        stage_destroy(&stage1);
        stage_destroy(&stage0);
        vip_destroy();
        return 1;
    }
    printf("Buffer sizes match, chain is valid.\n\n");

    /* Load initial input for stage0 */
    if (write_input_from_file(&stage0, block0_input) != 0) {
        stage_destroy(&stage1);
        stage_destroy(&stage0);
        vip_destroy();
        return 1;
    }
    printf("Loaded block0 input from %s\n\n", block0_input);

    printf("=== Running %d iterations of block0->block1 chain ===\n", iterations);
    uint64_t total_wall = 0;
    uint64_t min_wall = UINT64_MAX;
    uint64_t max_wall = 0;

    for (i = 0; i < iterations; i++) {
        vip_inference_profile_t profile0, profile1;
        uint64_t iter_start = now_us();

        /* Run block0 */
        if (stage_run(&stage0, 0, &profile0) != 0) {
            stage_destroy(&stage1);
            stage_destroy(&stage0);
            vip_destroy();
            return 1;
        }

        /* Chain: copy block0 output to block1 input */
        if (stage_copy_output_to_input(&stage0, &stage1) != 0) {
            stage_destroy(&stage1);
            stage_destroy(&stage0);
            vip_destroy();
            return 1;
        }

        /* Run block1 */
        if (stage_run(&stage1, 1, &profile1) != 0) {
            stage_destroy(&stage1);
            stage_destroy(&stage0);
            vip_destroy();
            return 1;
        }

        uint64_t iter_wall = now_us() - iter_start;
        total_wall += iter_wall;
        if (iter_wall < min_wall) min_wall = iter_wall;
        if (iter_wall > max_wall) max_wall = iter_wall;

        printf("[iter %d] chain_wall=%" PRIu64 "us (block0=%uus+block1=%uus)\n",
               i, iter_wall, profile0.inference_time, profile1.inference_time);
    }

    printf("\n--- Chain Timing Summary ---\n");
    printf("mean_chain_wall=%.1f us\n", (double)total_wall / (double)iterations);
    printf("min_chain_wall=%" PRIu64 " us\n", min_wall);
    printf("max_chain_wall=%" PRIu64 " us\n", max_wall);
    printf("nbg_loaded_once=1\n");
    printf("chain_iterations=%d\n", iterations);
    printf("vip_run ret=0\n");

    /* Cleanup */
    stage_destroy(&stage1);
    stage_destroy(&stage0);
    vip_destroy();

    printf("\n=== Chain test PASSED ===\n");
    return 0;
}
