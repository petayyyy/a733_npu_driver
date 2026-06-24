#define _POSIX_C_SOURCE 200809L

#include <vip_lite.h>

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#define MAX_PROMPT_TOKENS 256
#define MAX_TOPK 5

#ifdef A733_VIP_LEGACY_DEVICE_ID
#define A733_VIP_NETWORK_PROP_SET_DEVICE VIP_NETWORK_PROP_SET_DEVICE_ID
#else
#define A733_VIP_NETWORK_PROP_SET_DEVICE VIP_NETWORK_PROP_SET_DEVICE_INDEX
#endif

typedef struct {
    const char *model_dir;
    const char *nbg_path;
    const char *prompt;
    int steps;
    int seq_len;
    int vocab;
    int protocol;
    float temperature;
    unsigned int seed;
    uint32_t device_index;
    int32_t core_index;
    uint32_t timeout_ms;
} runner_args_t;

typedef struct {
    vip_network network;
    vip_buffer input;
    vip_buffer output;
    vip_buffer_create_params_t input_param;
    vip_buffer_create_params_t output_param;
    uint32_t input_elements;
    uint32_t output_elements;
    int prepared;
} runner_t;

static uint64_t now_us(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
}

static void usage(const char *argv0)
{
    fprintf(stderr,
            "Usage: %s [options]\n"
            "\n"
            "Persistent tiny-LM VIPLite runner. It loads network_binary.nb once,\n"
            "then repeatedly overwrites the int32 token window and runs the NPU.\n"
            "\n"
            "Options:\n"
            "  --model-dir DIR   Directory with network_binary.nb (default: cwd)\n"
            "  --nbg FILE        Explicit NBG path (overrides --model-dir)\n"
            "  --prompt IDS      Initial token IDs, comma or space separated (default: 1 5 9 2)\n"
            "  --steps N         Number of generated tokens (default: 8)\n"
            "  --seq-len N       Fixed input window length (default: 4)\n"
            "  --vocab N         Vocabulary size for last-position argmax (default: 16)\n"
            "  --protocol        Keep the NBG loaded and serve RUN commands over stdin/stdout\n"
            "  --temperature F   Sample with temperature F; <=0 means greedy argmax (default: 0)\n"
            "  --seed N          PRNG seed for temperature sampling (default: 1)\n"
            "  --device N        VIP device index (default: 0)\n"
            "  --core N          VIP core index; -1 keeps SDK default (default: -1)\n"
            "  --timeout-ms N    VIP network timeout in ms (default: 0)\n",
            argv0);
}

static int parse_int(const char *text, int *out)
{
    char *end = NULL;
    long value;

    errno = 0;
    value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < INT_MIN || value > INT_MAX) {
        return -1;
    }
    *out = (int)value;
    return 0;
}

static int parse_u32(const char *text, uint32_t *out)
{
    int value = 0;
    if (parse_int(text, &value) != 0 || value < 0) {
        return -1;
    }
    *out = (uint32_t)value;
    return 0;
}

static int parse_float(const char *text, float *out)
{
    char *end = NULL;
    double value;

    errno = 0;
    value = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(value)) {
        return -1;
    }
    *out = (float)value;
    return 0;
}

static int parse_args(int argc, char **argv, runner_args_t *args)
{
    int i;

    args->model_dir = ".";
    args->nbg_path = NULL;
    args->prompt = "1 5 9 2";
    args->steps = 8;
    args->seq_len = 4;
    args->vocab = 16;
    args->protocol = 0;
    args->temperature = 0.0f;
    args->seed = 1U;
    args->device_index = 0;
    args->core_index = -1;
    args->timeout_ms = 0;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
            args->model_dir = argv[++i];
        } else if (strcmp(argv[i], "--nbg") == 0 && i + 1 < argc) {
            args->nbg_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            args->prompt = argv[++i];
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &args->steps) != 0) {
                return -1;
            }
        } else if (strcmp(argv[i], "--seq-len") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &args->seq_len) != 0) {
                return -1;
            }
        } else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &args->vocab) != 0) {
                return -1;
            }
        } else if (strcmp(argv[i], "--protocol") == 0) {
            args->protocol = 1;
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            if (parse_float(argv[++i], &args->temperature) != 0) {
                return -1;
            }
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            uint32_t seed = 0;
            if (parse_u32(argv[++i], &seed) != 0) {
                return -1;
            }
            args->seed = seed;
        } else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            if (parse_u32(argv[++i], &args->device_index) != 0) {
                return -1;
            }
        } else if (strcmp(argv[i], "--core") == 0 && i + 1 < argc) {
            int core = 0;
            if (parse_int(argv[++i], &core) != 0) {
                return -1;
            }
            args->core_index = (int32_t)core;
        } else if (strcmp(argv[i], "--timeout-ms") == 0 && i + 1 < argc) {
            if (parse_u32(argv[++i], &args->timeout_ms) != 0) {
                return -1;
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            exit(0);
        } else {
            return -1;
        }
    }

    if (args->steps < 0 || args->seq_len <= 0 || args->vocab <= 0) {
        return -1;
    }
    if (args->seq_len > MAX_PROMPT_TOKENS) {
        fprintf(stderr, "seq-len must be <= %d\n", MAX_PROMPT_TOKENS);
        return -1;
    }
    if (!args->protocol && args->steps > MAX_PROMPT_TOKENS - args->seq_len) {
        fprintf(stderr, "seq-len + steps must be <= %d\n", MAX_PROMPT_TOKENS);
        return -1;
    }
    return 0;
}

static void *read_file(const char *path, uint32_t *size_out)
{
    FILE *fp = NULL;
    long size = 0;
    void *data = NULL;

    fp = fopen(path, "rb");
    if (fp == NULL) {
        fprintf(stderr, "failed to open %s: %s\n", path, strerror(errno));
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "failed to seek %s\n", path);
        goto fail;
    }
    size = ftell(fp);
    if (size <= 0 || size > UINT32_MAX) {
        fprintf(stderr, "invalid NBG size for %s: %ld\n", path, size);
        goto fail;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "failed to rewind %s\n", path);
        goto fail;
    }

    data = malloc((size_t)size);
    if (data == NULL) {
        fprintf(stderr, "failed to allocate %ld bytes for %s\n", size, path);
        goto fail;
    }
    if (fread(data, 1, (size_t)size, fp) != (size_t)size) {
        fprintf(stderr, "failed to read %s\n", path);
        free(data);
        data = NULL;
        goto fail;
    }

    fclose(fp);
    *size_out = (uint32_t)size;
    return data;

fail:
    fclose(fp);
    return NULL;
}

static uint32_t type_bytes(vip_enum type)
{
    switch (type) {
    case VIP_BUFFER_FORMAT_INT8:
    case VIP_BUFFER_FORMAT_UINT8:
    case VIP_BUFFER_FORMAT_BOOL8:
        return 1;
    case VIP_BUFFER_FORMAT_INT16:
    case VIP_BUFFER_FORMAT_UINT16:
    case VIP_BUFFER_FORMAT_FP16:
    case VIP_BUFFER_FORMAT_BFP16:
        return 2;
    case VIP_BUFFER_FORMAT_FP32:
    case VIP_BUFFER_FORMAT_INT32:
    case VIP_BUFFER_FORMAT_UINT32:
        return 4;
    case VIP_BUFFER_FORMAT_FP64:
    case VIP_BUFFER_FORMAT_INT64:
    case VIP_BUFFER_FORMAT_UINT64:
        return 8;
    default:
        return 0;
    }
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

static float fp16_to_fp32(uint16_t in)
{
    uint32_t sign = ((uint32_t)in & 0x8000U) << 16;
    uint32_t exp = ((uint32_t)in >> 10) & 0x1FU;
    uint32_t mant = (uint32_t)in & 0x03FFU;
    uint32_t out;
    float result;

    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400U) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FFU;
            out = sign | ((exp + 127U - 15U) << 23) | (mant << 13);
        }
    } else if (exp == 0x1FU) {
        out = sign | 0x7F800000U | (mant << 13);
    } else {
        out = sign | ((exp + 127U - 15U) << 23) | (mant << 13);
    }

    memcpy(&result, &out, sizeof(result));
    return result;
}

static int64_t read_integer(const uint8_t *ptr, vip_enum type)
{
    switch (type) {
    case VIP_BUFFER_FORMAT_INT8:
        return *(const int8_t *)ptr;
    case VIP_BUFFER_FORMAT_UINT8:
    case VIP_BUFFER_FORMAT_BOOL8:
        return *(const uint8_t *)ptr;
    case VIP_BUFFER_FORMAT_INT16:
        return *(const int16_t *)ptr;
    case VIP_BUFFER_FORMAT_UINT16:
        return *(const uint16_t *)ptr;
    case VIP_BUFFER_FORMAT_INT32:
        return *(const int32_t *)ptr;
    case VIP_BUFFER_FORMAT_UINT32:
        return *(const uint32_t *)ptr;
    case VIP_BUFFER_FORMAT_INT64:
        return *(const int64_t *)ptr;
    case VIP_BUFFER_FORMAT_UINT64:
        return (int64_t)*(const uint64_t *)ptr;
    default:
        return 0;
    }
}

static float decode_value(const uint8_t *ptr, const vip_buffer_create_params_t *param)
{
    int64_t raw;
    float value;

    if (param->data_format == VIP_BUFFER_FORMAT_FP32) {
        memcpy(&value, ptr, sizeof(value));
        return value;
    }
    if (param->data_format == VIP_BUFFER_FORMAT_FP16) {
        uint16_t half = 0;
        memcpy(&half, ptr, sizeof(half));
        return fp16_to_fp32(half);
    }

    raw = read_integer(ptr, param->data_format);
    if (param->quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) {
        int fl = param->quant_data.dfp.fixed_point_pos;
        if (fl >= 0) {
            return (float)raw / (float)(1ULL << fl);
        }
        return (float)raw * (float)(1ULL << (-fl));
    }
    if (param->quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM) {
        return ((float)raw - (float)param->quant_data.affine.zeroPoint) *
               param->quant_data.affine.scale;
    }
    return (float)raw;
}

static int parse_prompt(const char *text, int vocab, int *tokens, int *count)
{
    char *copy = NULL;
    char *save = NULL;
    char *part = NULL;
    int n = 0;

    copy = malloc(strlen(text) + 1);
    if (copy == NULL) {
        return -1;
    }
    strcpy(copy, text);

    for (part = strtok_r(copy, " ,\t\r\n", &save);
         part != NULL;
         part = strtok_r(NULL, " ,\t\r\n", &save)) {
        int token = 0;
        if (n >= MAX_PROMPT_TOKENS || parse_int(part, &token) != 0 ||
            token < 0 || token >= vocab) {
            free(copy);
            return -1;
        }
        tokens[n++] = token;
    }

    free(copy);
    if (n == 0) {
        return -1;
    }
    *count = n;
    return 0;
}

static void print_window(const int *tokens, int start, int seq_len)
{
    int i;
    for (i = 0; i < seq_len; i++) {
        printf("%s%d", i == 0 ? "" : " ", tokens[start + i]);
    }
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
    if (status != VIP_SUCCESS) {
        fprintf(stderr, "vip_create_buffer(%s) failed: %d\n",
                is_input ? "input" : "output", status);
    }
    return status;
}

static int runner_create(runner_t *runner, const runner_args_t *args, const char *nbg_path)
{
    void *nbg_data = NULL;
    uint32_t nbg_size = 0;
    vip_status_e status;
    uint32_t input_count = 0;
    uint32_t output_count = 0;
    uint32_t mem_pool_size = 0;
    uint8_t core_count = 0;
    uint64_t t0;

    memset(runner, 0, sizeof(*runner));
    nbg_data = read_file(nbg_path, &nbg_size);
    if (nbg_data == NULL) {
        return -1;
    }

    t0 = now_us();
    status = vip_create_network(nbg_data, nbg_size, VIP_CREATE_NETWORK_FROM_MEMORY, &runner->network);
    free(nbg_data);
    if (status != VIP_SUCCESS) {
        fprintf(stderr, "vip_create_network failed: %d\n", status);
        return -1;
    }
    printf("nbg_path=%s\n", nbg_path);
    printf("nbg_size=%u\n", nbg_size);
    printf("create_network_us=%" PRIu64 "\n", now_us() - t0);

    if (args->device_index > 0) {
        status = vip_set_network(runner->network, A733_VIP_NETWORK_PROP_SET_DEVICE,
                                 (void *)&args->device_index);
        if (status != VIP_SUCCESS) {
            fprintf(stderr, "vip_set_network(device) failed: %d\n", status);
            return -1;
        }
    }
    if (args->core_index != -1) {
#ifndef A733_VIP_NO_CORE_INDEX
        status = vip_set_network(runner->network, VIP_NETWORK_PROP_SET_CORE_INDEX,
                                 (void *)&args->core_index);
        if (status != VIP_SUCCESS) {
            fprintf(stderr, "vip_set_network(core) failed: %d\n", status);
            return -1;
        }
#else
        fprintf(stderr, "vip_set_network(core) skipped: SDK has no core-index property\n");
#endif
    }
    if (args->timeout_ms != 0) {
        status = vip_set_network(runner->network, VIP_NETWORK_PROP_SET_TIME_OUT,
                                 (void *)&args->timeout_ms);
        if (status != VIP_SUCCESS) {
            fprintf(stderr, "vip_set_network(timeout) failed: %d\n", status);
            return -1;
        }
    }

    vip_query_network(runner->network, VIP_NETWORK_PROP_INPUT_COUNT, &input_count);
    vip_query_network(runner->network, VIP_NETWORK_PROP_OUTPUT_COUNT, &output_count);
    if (input_count != 1 || output_count != 1) {
        fprintf(stderr, "expected one input and one output, got %u/%u\n", input_count, output_count);
        return -1;
    }

    if (create_buffer_from_query(runner->network, 1, &runner->input_param, &runner->input) !=
        VIP_SUCCESS) {
        return -1;
    }
    if (create_buffer_from_query(runner->network, 0, &runner->output_param, &runner->output) !=
        VIP_SUCCESS) {
        return -1;
    }

    runner->input_elements = element_count(&runner->input_param);
    runner->output_elements = element_count(&runner->output_param);

    printf("input_dims=");
    print_dims(&runner->input_param);
    printf(" input_format=%d input_quant=%d input_elements=%u input_bytes=%u\n",
           runner->input_param.data_format, runner->input_param.quant_format,
           runner->input_elements, vip_get_buffer_size(runner->input));
    printf("output_dims=");
    print_dims(&runner->output_param);
    printf(" output_format=%d output_quant=%d output_dfp=%d output_elements=%u output_bytes=%u\n",
           runner->output_param.data_format, runner->output_param.quant_format,
           runner->output_param.quant_data.dfp.fixed_point_pos, runner->output_elements,
           vip_get_buffer_size(runner->output));

    if (runner->input_param.data_format != VIP_BUFFER_FORMAT_INT32 ||
        runner->input_elements < (uint32_t)args->seq_len) {
        fprintf(stderr, "expected int32 input with at least seq_len elements\n");
        return -1;
    }
    if (runner->output_elements < (uint32_t)args->vocab ||
        runner->output_elements % (uint32_t)args->vocab != 0) {
        fprintf(stderr, "output elements %u are not compatible with vocab %d\n",
                runner->output_elements, args->vocab);
        return -1;
    }

    vip_query_network(runner->network, VIP_NETWORK_PROP_MEMORY_POOL_SIZE, &mem_pool_size);
    vip_query_network(runner->network, VIP_NETWORK_PROP_CORE_COUNT, &core_count);
    printf("memory_pool_bytes=%u\n", mem_pool_size);
    printf("network_core_count=%u\n", (unsigned)core_count);

    t0 = now_us();
    status = vip_prepare_network(runner->network);
    if (status != VIP_SUCCESS) {
        fprintf(stderr, "vip_prepare_network failed: %d\n", status);
        return -1;
    }
    runner->prepared = 1;
    printf("prepare_network_us=%" PRIu64 "\n", now_us() - t0);

    status = vip_set_input(runner->network, 0, runner->input);
    if (status != VIP_SUCCESS) {
        fprintf(stderr, "vip_set_input failed: %d\n", status);
        return -1;
    }
    status = vip_set_output(runner->network, 0, runner->output);
    if (status != VIP_SUCCESS) {
        fprintf(stderr, "vip_set_output failed: %d\n", status);
        return -1;
    }

    return 0;
}

static void runner_destroy(runner_t *runner)
{
    if (runner->prepared && runner->network != NULL) {
        vip_finish_network(runner->network);
    }
    if (runner->input != NULL) {
        vip_destroy_buffer(runner->input);
    }
    if (runner->output != NULL) {
        vip_destroy_buffer(runner->output);
    }
    if (runner->network != NULL) {
        vip_destroy_network(runner->network);
    }
    memset(runner, 0, sizeof(*runner));
}

static int write_window(runner_t *runner, const int *window, int seq_len)
{
    int32_t *mapped = (int32_t *)vip_map_buffer(runner->input);
    uint32_t input_size = vip_get_buffer_size(runner->input);

    if (mapped == NULL) {
        fprintf(stderr, "vip_map_buffer(input) failed\n");
        return -1;
    }
    if (input_size < (uint32_t)seq_len * sizeof(int32_t)) {
        fprintf(stderr, "input buffer too small: %u\n", input_size);
        vip_unmap_buffer(runner->input);
        return -1;
    }
    memcpy(mapped, window, (size_t)seq_len * sizeof(int32_t));
    vip_unmap_buffer(runner->input);

    if (vip_flush_buffer(runner->input, VIP_BUFFER_OPER_TYPE_FLUSH) != VIP_SUCCESS) {
        fprintf(stderr, "vip_flush_buffer(input) failed\n");
        return -1;
    }
    return 0;
}

static int read_next_token(
    runner_t *runner,
    int vocab,
    float temperature,
    char *top5,
    size_t top5_size)
{
    uint8_t *mapped = NULL;
    uint32_t stride = type_bytes(runner->output_param.data_format);
    uint32_t base = runner->output_elements - (uint32_t)vocab;
    float best = 0.0f;
    int best_idx = 0;
    int selected_idx = 0;
    int top_idx[MAX_TOPK];
    float top_val[MAX_TOPK];
    int i;
    int k;
    size_t used = 0;

    if (stride == 0) {
        fprintf(stderr, "unsupported output data format: %d\n", runner->output_param.data_format);
        return -1;
    }
    if (vip_flush_buffer(runner->output, VIP_BUFFER_OPER_TYPE_INVALIDATE) != VIP_SUCCESS) {
        fprintf(stderr, "vip_flush_buffer(output) failed\n");
        return -1;
    }

    mapped = (uint8_t *)vip_map_buffer(runner->output);
    if (mapped == NULL) {
        fprintf(stderr, "vip_map_buffer(output) failed\n");
        return -1;
    }

    for (k = 0; k < MAX_TOPK; k++) {
        top_idx[k] = -1;
        top_val[k] = -3.402823466e+38F;
    }

    for (i = 0; i < vocab; i++) {
        float value = decode_value(mapped + ((base + (uint32_t)i) * stride), &runner->output_param);
        if (i == 0 || value > best) {
            best = value;
            best_idx = i;
        }
        for (k = 0; k < MAX_TOPK; k++) {
            if (value > top_val[k]) {
                int move;
                for (move = MAX_TOPK - 1; move > k; move--) {
                    top_val[move] = top_val[move - 1];
                    top_idx[move] = top_idx[move - 1];
                }
                top_val[k] = value;
                top_idx[k] = i;
                break;
            }
        }
    }

    selected_idx = best_idx;
    if (temperature > 0.0f) {
        double sum = 0.0;
        double draw;
        double cumulative = 0.0;

        for (i = 0; i < vocab; i++) {
            float value = decode_value(mapped + ((base + (uint32_t)i) * stride),
                                       &runner->output_param);
            double weight = exp(((double)value - (double)best) / (double)temperature);
            if (isfinite(weight) && weight > 0.0) {
                sum += weight;
            }
        }

        if (isfinite(sum) && sum > 0.0) {
            draw = ((double)rand() / ((double)RAND_MAX + 1.0)) * sum;
            for (i = 0; i < vocab; i++) {
                float value = decode_value(mapped + ((base + (uint32_t)i) * stride),
                                           &runner->output_param);
                double weight = exp(((double)value - (double)best) / (double)temperature);
                if (isfinite(weight) && weight > 0.0) {
                    cumulative += weight;
                    if (cumulative >= draw) {
                        selected_idx = i;
                        break;
                    }
                }
            }
        }
    }

    vip_unmap_buffer(runner->output);

    for (k = 0; k < MAX_TOPK && k < vocab; k++) {
        int wrote = snprintf(top5 + used, top5_size > used ? top5_size - used : 0,
                             "%s%d:%.6f", k == 0 ? "" : ",", top_idx[k], top_val[k]);
        if (wrote < 0) {
            break;
        }
        used += (size_t)wrote;
        if (used >= top5_size) {
            break;
        }
    }

    return selected_idx;
}

static int run_window_once(
    runner_t *runner,
    const runner_args_t *args,
    const int *window,
    int *next_token,
    char *top5,
    size_t top5_size,
    uint32_t *profile_us,
    uint32_t *cycle,
    uint64_t *wall_us)
{
    uint64_t t0;
    vip_status_e status;
    vip_inference_profile_t profile;

    t0 = now_us();
    if (write_window(runner, window, args->seq_len) != 0) {
        return -1;
    }

    status = vip_run_network(runner->network);
    if (status != VIP_SUCCESS) {
        fprintf(stderr, "vip_run_network failed: %d\n", status);
        return -1;
    }

    memset(&profile, 0, sizeof(profile));
    vip_query_network(runner->network, VIP_NETWORK_PROP_PROFILING, &profile);
    *next_token = read_next_token(runner, args->vocab, args->temperature, top5, top5_size);
    if (*next_token < 0) {
        return -1;
    }

    *wall_us = now_us() - t0;
    *profile_us = profile.inference_time;
    *cycle = profile.total_cycle;
    return 0;
}

static int run_decode(runner_t *runner, const runner_args_t *args, int *tokens, int token_count)
{
    int step;
    uint64_t total_wall_us = 0;
    uint64_t total_profile_us = 0;

    printf("initial_tokens=");
    print_window(tokens, 0, token_count);
    printf("\n");
    printf("nbg_loaded_once=1\n");

    for (step = 0; step < args->steps; step++) {
        int window[MAX_PROMPT_TOKENS];
        int start = token_count - args->seq_len;
        int next_token;
        int i;
        uint64_t wall_us;
        uint32_t profile_us;
        uint32_t cycle;
        char top5[256];

        for (i = 0; i < args->seq_len; i++) {
            window[i] = tokens[start + i];
        }

        if (run_window_once(runner, args, window, &next_token, top5, sizeof(top5),
                            &profile_us, &cycle, &wall_us) != 0) {
            return -1;
        }
        total_wall_us += wall_us;
        total_profile_us += profile_us;

        printf("step=%d window=", step);
        print_window(window, 0, args->seq_len);
        printf(" next=%d top5=%s profile_us=%u cycle=%u wall_us=%" PRIu64 "\n",
               next_token, top5, profile_us, cycle, wall_us);

        tokens[token_count++] = next_token;
    }

    printf("final_tokens=");
    print_window(tokens, 0, token_count);
    printf("\n");
    if (args->steps > 0) {
        double mean_wall_us = (double)total_wall_us / (double)args->steps;
        double mean_profile_us = (double)total_profile_us / (double)args->steps;
        printf("mean_wall_us=%.3f\n", mean_wall_us);
        printf("mean_profile_us=%.3f\n", mean_profile_us);
        printf("mean_tok_s=%.3f\n", 1000000.0 / mean_wall_us);
    }
    return 0;
}

static int parse_protocol_window(const char *text, int vocab, int seq_len, int *window)
{
    char *copy = NULL;
    char *save = NULL;
    char *part = NULL;
    int n = 0;

    copy = malloc(strlen(text) + 1);
    if (copy == NULL) {
        return -1;
    }
    strcpy(copy, text);

    for (part = strtok_r(copy, " ,\t\r\n", &save);
         part != NULL;
         part = strtok_r(NULL, " ,\t\r\n", &save)) {
        int token = 0;
        if (n >= seq_len || parse_int(part, &token) != 0 || token < 0 || token >= vocab) {
            free(copy);
            return -1;
        }
        window[n++] = token;
    }

    free(copy);
    return n == seq_len ? 0 : -1;
}

static int run_protocol(runner_t *runner, const runner_args_t *args)
{
    char line[8192];

    printf("protocol=stdio\n");
    printf("nbg_loaded_once=1\n");
    printf("READY seq_len=%d vocab=%d temperature=%.6g\n",
           args->seq_len, args->vocab, args->temperature);
    fflush(stdout);

    while (fgets(line, sizeof(line), stdin) != NULL) {
        char *command = line;
        while (*command == ' ' || *command == '\t') {
            command++;
        }

        if (strncmp(command, "QUIT", 4) == 0 || strncmp(command, "EXIT", 4) == 0) {
            printf("BYE\n");
            fflush(stdout);
            return 0;
        }
        if (strncmp(command, "RUN", 3) == 0 &&
            (command[3] == '\0' || command[3] == ' ' || command[3] == '\t')) {
            int window[MAX_PROMPT_TOKENS];
            int next_token = -1;
            uint32_t profile_us = 0;
            uint32_t cycle = 0;
            uint64_t wall_us = 0;
            char top5[256];
            char *payload = command + 3;

            if (parse_protocol_window(payload, args->vocab, args->seq_len, window) != 0) {
                printf("ERROR invalid_window expected=%d\n", args->seq_len);
                fflush(stdout);
                continue;
            }
            if (run_window_once(runner, args, window, &next_token, top5, sizeof(top5),
                                &profile_us, &cycle, &wall_us) != 0) {
                printf("ERROR run_failed\n");
                fflush(stdout);
                return -1;
            }
            printf("TOKEN id=%d profile_us=%u cycle=%u wall_us=%" PRIu64 " top5=%s\n",
                   next_token, profile_us, cycle, wall_us, top5);
            fflush(stdout);
            continue;
        }

        printf("ERROR unknown_command\n");
        fflush(stdout);
    }

    return 0;
}

int main(int argc, char **argv)
{
    runner_args_t args;
    runner_t runner;
    char default_nbg[PATH_MAX];
    const char *nbg_path;
    int tokens[MAX_PROMPT_TOKENS];
    int prompt_count = 0;
    int token_count = 0;
    int i;
    vip_status_e status;
    uint32_t version;
    uint32_t cid = 0;
    uint32_t device_count = 0;
    int ret = 1;

    if (parse_args(argc, argv, &args) != 0) {
        usage(argv[0]);
        return 2;
    }

    if (args.nbg_path != NULL) {
        nbg_path = args.nbg_path;
    } else {
        snprintf(default_nbg, sizeof(default_nbg), "%s/%s", args.model_dir, "network_binary.nb");
        nbg_path = default_nbg;
    }

    if (!args.protocol) {
        if (parse_prompt(args.prompt, args.vocab, tokens, &prompt_count) != 0) {
            fprintf(stderr, "invalid prompt: %s\n", args.prompt);
            return 2;
        }
        if (prompt_count > args.seq_len) {
            fprintf(stderr, "prompt has %d tokens, seq-len is %d\n", prompt_count, args.seq_len);
            return 2;
        }

        token_count = args.seq_len;
        memmove(tokens + (args.seq_len - prompt_count), tokens,
                (size_t)prompt_count * sizeof(tokens[0]));
        for (i = 0; i < args.seq_len - prompt_count; i++) {
            tokens[i] = 0;
        }
    }
    srand(args.seed);

    version = vip_get_version();
    printf("vip_lite_driver_version=0x%08x\n", version);

    status = vip_init();
    if (status != VIP_SUCCESS) {
        fprintf(stderr, "vip_init failed: %d\n", status);
        return 1;
    }
    printf("vip_init=OK\n");

    if (vip_query_hardware(VIP_QUERY_HW_PROP_CID, sizeof(cid), &cid) == VIP_SUCCESS) {
        printf("cid=0x%08x\n", cid);
    }
    if (vip_query_hardware(VIP_QUERY_HW_PROP_DEVICE_COUNT, sizeof(device_count), &device_count) ==
        VIP_SUCCESS) {
        printf("device_count=%u\n", device_count);
    }

    if (runner_create(&runner, &args, nbg_path) != 0) {
        goto out;
    }
    if (args.protocol) {
        if (run_protocol(&runner, &args) != 0) {
            goto out_destroy_runner;
        }
    } else if (run_decode(&runner, &args, tokens, token_count) != 0) {
        goto out_destroy_runner;
    }

    ret = 0;

out_destroy_runner:
    runner_destroy(&runner);
out:
    status = vip_destroy();
    if (status != VIP_SUCCESS) {
        fprintf(stderr, "vip_destroy failed: %d\n", status);
        ret = 1;
    }
    return ret;
}
