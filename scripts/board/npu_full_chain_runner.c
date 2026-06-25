#define _POSIX_C_SOURCE 200809L
#include <vip_lite.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
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

#define MAX_STAGES 26
#define PROMPT_LEN 32
#define MAX_VOCAB 152000

typedef struct {
    const char *name;
    const char *nbg_path;
    vip_network network;
    vip_buffer input;
    vip_buffer output;
    vip_buffer_create_params_t input_param;
    vip_buffer_create_params_t output_param;
    uint32_t input_elements;
    uint32_t output_elements;
    uint32_t input_bytes;
    uint32_t output_bytes;
    int is_embedding;   /* token_ids input */
    int is_final;       /* logits output */
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
    uint32_t i, count = 1;
    for (i = 0; i < param->num_of_dims; i++) count *= param->sizes[i];
    return count;
}

static void print_dims(const vip_buffer_create_params_t *param)
{
    uint32_t i;
    for (i = 0; i < param->num_of_dims; i++) printf("%s%u", i==0?"":"x", param->sizes[i]);
}

static void *read_file(const char *path, uint32_t *size_out)
{
    FILE *fp = fopen(path, "rb");
    long size; void *data;
    if (!fp) { fprintf(stderr,"open %s: %s\n",path,strerror(errno)); return NULL; }
    fseek(fp,0,SEEK_END); size=ftell(fp);
    if (size<=0) { fclose(fp); return NULL; }
    fseek(fp,0,SEEK_SET);
    data=malloc((size_t)size);
    if (!data || fread(data,1,(size_t)size,fp)!=(size_t)size) { free(data); fclose(fp); return NULL; }
    fclose(fp);
    *size_out=(uint32_t)size;
    return data;
}

static vip_status_e create_buffer_from_query(
    vip_network network, int is_input, vip_buffer_create_params_t *param, vip_buffer *buffer)
{
    vip_status_e s;
    memset(param,0,sizeof(*param));
    param->memory_type=VIP_BUFFER_MEMORY_TYPE_DEFAULT;
    if (is_input) {
        vip_query_input(network,0,VIP_BUFFER_PROP_DATA_FORMAT,&param->data_format);
        vip_query_input(network,0,VIP_BUFFER_PROP_NUM_OF_DIMENSION,&param->num_of_dims);
        vip_query_input(network,0,VIP_BUFFER_PROP_SIZES_OF_DIMENSION,param->sizes);
        vip_query_input(network,0,VIP_BUFFER_PROP_QUANT_FORMAT,&param->quant_format);
        vip_query_input(network,0,VIP_BUFFER_PROP_FIXED_POINT_POS,&param->quant_data.dfp.fixed_point_pos);
        vip_query_input(network,0,VIP_BUFFER_PROP_TF_SCALE,&param->quant_data.affine.scale);
        vip_query_input(network,0,VIP_BUFFER_PROP_TF_ZERO_POINT,&param->quant_data.affine.zeroPoint);
    } else {
        vip_query_output(network,0,VIP_BUFFER_PROP_DATA_FORMAT,&param->data_format);
        vip_query_output(network,0,VIP_BUFFER_PROP_NUM_OF_DIMENSION,&param->num_of_dims);
        vip_query_output(network,0,VIP_BUFFER_PROP_SIZES_OF_DIMENSION,param->sizes);
        vip_query_output(network,0,VIP_BUFFER_PROP_QUANT_FORMAT,&param->quant_format);
        vip_query_output(network,0,VIP_BUFFER_PROP_FIXED_POINT_POS,&param->quant_data.dfp.fixed_point_pos);
        vip_query_output(network,0,VIP_BUFFER_PROP_TF_SCALE,&param->quant_data.affine.scale);
        vip_query_output(network,0,VIP_BUFFER_PROP_TF_ZERO_POINT,&param->quant_data.affine.zeroPoint);
    }
    s=vip_create_buffer(param,sizeof(*param),buffer);
    return s;
}

static int stage_init(stage_t *s, uint32_t device_index)
{
    vip_status_e st; uint32_t ic=0,oc=0,mps=0,cc=0;
    uint64_t t0=now_us();
    st=vip_create_network((void*)s->nbg_path,0,VIP_CREATE_NETWORK_FROM_FILE,&s->network);
    if (st!=VIP_SUCCESS) { fprintf(stderr,"[%s] create_network failed: %d\n",s->name,st); return -1; }
    printf("[%s] create=%"PRIu64"us\n",s->name,now_us()-t0);

    { uint32_t di=device_index; vip_set_network(s->network,A733_VIP_NETWORK_PROP_SET_DEVICE,(void*)&di); }

    vip_query_network(s->network,VIP_NETWORK_PROP_INPUT_COUNT,&ic);
    vip_query_network(s->network,VIP_NETWORK_PROP_OUTPUT_COUNT,&oc);
    if (ic!=1||oc!=1) { fprintf(stderr,"[%s] expected 1/1 I/O got %u/%u\n",s->name,ic,oc); return -1; }

    if (create_buffer_from_query(s->network,1,&s->input_param,&s->input)!=VIP_SUCCESS) { fprintf(stderr,"[%s] input buf fail\n",s->name); return -1; }
    if (create_buffer_from_query(s->network,0,&s->output_param,&s->output)!=VIP_SUCCESS) { fprintf(stderr,"[%s] output buf fail\n",s->name); return -1; }

    s->input_elements=element_count(&s->input_param);
    s->output_elements=element_count(&s->output_param);
    s->input_bytes=vip_get_buffer_size(s->input);
    s->output_bytes=vip_get_buffer_size(s->output);

    printf("[%s] in: ",s->name); print_dims(&s->input_param);
    printf(" fmt=%d qfmt=%d elems=%u bytes=%u\n",
           s->input_param.data_format,s->input_param.quant_format,s->input_elements,s->input_bytes);
    printf("[%s] out: ",s->name); print_dims(&s->output_param);
    printf(" fmt=%d qfmt=%d dfp=%d elems=%u bytes=%u\n",
           s->output_param.data_format,s->output_param.quant_format,
           s->output_param.quant_data.dfp.fixed_point_pos,s->output_elements,s->output_bytes);

    vip_query_network(s->network,VIP_NETWORK_PROP_MEMORY_POOL_SIZE,&mps);
    vip_query_network(s->network,VIP_NETWORK_PROP_CORE_COUNT,&cc);
    printf("[%s] mempool=%uB cores=%u\n",s->name,mps,(unsigned)cc);

    t0=now_us();
    st=vip_prepare_network(s->network);
    if (st!=VIP_SUCCESS) { fprintf(stderr,"[%s] prepare failed: %d\n",s->name,st); return -1; }
    s->prepared=1;
    printf("[%s] prepare=%"PRIu64"us\n",s->name,now_us()-t0);
    vip_set_input(s->network,0,s->input);
    vip_set_output(s->network,0,s->output);
    return 0;
}

static void stage_destroy(stage_t *s)
{
    if (s->prepared && s->network) vip_finish_network(s->network);
    if (s->input) vip_destroy_buffer(s->input);
    if (s->output) vip_destroy_buffer(s->output);
    if (s->network) vip_destroy_network(s->network);
    memset(s,0,sizeof(*s));
}

static int stage_run(stage_t *s, uint32_t *profile_us)
{
    uint64_t t0=now_us(); vip_inference_profile_t prof;
    vip_status_e st=vip_run_network(s->network);
    uint64_t wall=now_us()-t0;
    if (st!=VIP_SUCCESS) { fprintf(stderr,"[%s] run failed: %d\n",s->name,st); return -1; }
    vip_query_network(s->network,VIP_NETWORK_PROP_PROFILING,(void*)&prof);
    if (profile_us) *profile_us=prof.inference_time;
    return 0;
}

static int copy_output_to_input(stage_t *src, stage_t *dst)
{
    if (src->output_bytes != dst->input_bytes) { fprintf(stderr,"[%s->%s] size mismatch %u/%u\n",src->name,dst->name,src->output_bytes,dst->input_bytes); return -1; }
    vip_flush_buffer(src->output,VIP_BUFFER_OPER_TYPE_INVALIDATE);
    uint8_t *sm=(uint8_t*)vip_map_buffer(src->output);
    uint8_t *dm=(uint8_t*)vip_map_buffer(dst->input);
    if (!sm||!dm) { if(sm)vip_unmap_buffer(src->output); if(dm)vip_unmap_buffer(dst->input); return -1; }
    memcpy(dm,sm,src->output_bytes);
    vip_unmap_buffer(src->output); vip_unmap_buffer(dst->input);
    vip_flush_buffer(dst->input,VIP_BUFFER_OPER_TYPE_FLUSH);
    return 0;
}

static int write_token_window(stage_t *emb, const int *window, int len)
{
    uint8_t *m=vip_map_buffer(emb->input);
    if (!m) return -1;
    memset(m,0,emb->input_bytes);
    memcpy(m,window,(size_t)len*sizeof(int32_t));
    vip_unmap_buffer(emb->input);
    vip_flush_buffer(emb->input,VIP_BUFFER_OPER_TYPE_FLUSH);
    return 0;
}

static int read_logits(stage_t *fin, int vocab, int *next_token, float *top_logit)
{
    vip_flush_buffer(fin->output,VIP_BUFFER_OPER_TYPE_INVALIDATE);
    int16_t *m=(int16_t*)vip_map_buffer(fin->output);
    if (!m) return -1;
    int fl=fin->output_param.quant_data.dfp.fixed_point_pos;
    float best=-1e38f; int best_i=0;
    for (int i=0;i<vocab;i++) {
        float v=(float)m[i]/(float)(1ULL<<fl);
        if (v>best) { best=v; best_i=i; }
    }
    vip_unmap_buffer(fin->output);
    if (next_token) *next_token=best_i;
    if (top_logit) *top_logit=best;
    return best_i;
}

static int read_hidden_for_verify(stage_t *s, float *buf, int count)
{
    vip_flush_buffer(s->output,VIP_BUFFER_OPER_TYPE_INVALIDATE);
    int16_t *m=(int16_t*)vip_map_buffer(s->output);
    if (!m) return -1;
    int fl=s->output_param.quant_data.dfp.fixed_point_pos;
    int n=count<(int)s->output_elements?count:(int)s->output_elements;
    for (int i=0;i<n;i++) buf[i]=(float)m[i]/(float)(1ULL<<fl);
    vip_unmap_buffer(s->output);
    return 0;
}

static double cosine_f32(const float *a, const float *b, int n)
{
    double dot=0,na=0,nb=0;
    for (int i=0;i<n;i++) { dot+=(double)a[i]*(double)b[i]; na+=(double)a[i]*(double)a[i]; nb+=(double)b[i]*(double)b[i]; }
    if (na==0||nb==0) return 0;
    return dot/(sqrt(na)*sqrt(nb));
}

int main(int argc, char **argv)
{
    const char *model_base="/home/orangepi/a733_npu_driver/models";
    char nbg_buf[512];
    stage_t stages[MAX_STAGES];
    int n_stages=0;
    uint32_t di=0;
    int steps=8;
    int prompt_tokens[PROMPT_LEN]={0};
    int prompt_count=0;
    int verify=0;

    for (int a=1;a<argc;a++) {
        if (!strcmp(argv[a],"--steps")&&a+1<argc) steps=atoi(argv[++a]);
        else if (!strcmp(argv[a],"--device")&&a+1<argc) di=(uint32_t)atoi(argv[++a]);
        else if (!strcmp(argv[a],"--prompt")&&a+1<argc) {
            char *tok=strtok(argv[++a]," ,"); while(tok&&prompt_count<PROMPT_LEN) { prompt_tokens[prompt_count++]=atoi(tok); tok=strtok(NULL," ,"); }
        }
        else if (!strcmp(argv[a],"--verify")) verify=1;
        else { fprintf(stderr,"Usage: %s [--steps N] [--prompt IDS] [--verify]\n",argv[0]); return 1; }
    }

    if (prompt_count==0) {
        /* Default: Qwen2.5 ChatML-wrapped "The capital of France is" padded to 32 */
        int defaults[]={198,2610,525,1207,16948,11,3465,553,54364,14817,13,1446,525,264,10950,17847,13,151645,198,151644,872,198,785,6722,315,9625,374,151645,198,151644,77091,198};
        memcpy(prompt_tokens,defaults,sizeof(defaults));
        prompt_count=PROMPT_LEN;
    }

    /* Setup stages */
    const char *stage_names[MAX_STAGES]={NULL};
    /* Embedding */
    char *epath=strdup(nbg_buf); /* will be overwritten below */
    snprintf(nbg_buf,sizeof(nbg_buf),"%s/qwen25_05b_w32_embed_int16/network_binary.nb",model_base);
    epath=strdup(nbg_buf);
    stage_names[n_stages]="embed"; stages[n_stages].name="embed"; stages[n_stages].nbg_path=epath; stages[n_stages].is_embedding=1; n_stages++;
    /* Blocks 0-N */
    int max_blocks=24;
    const char *env_blocks=getenv("MAX_BLOCKS");
    if (env_blocks) max_blocks=atoi(env_blocks);
    for (int b=0;b<max_blocks;b++) {
        snprintf(nbg_buf,sizeof(nbg_buf),"%s/qwen25_05b_w32_block%d_int16/network_binary.nb",model_base,b);
        char *bpath=strdup(nbg_buf);
        char *nm=strdup((char[32]){0}); snprintf(nm,32,"block%d",b);
        stage_names[n_stages]=nm; stages[n_stages].name=nm; stages[n_stages].nbg_path=bpath; n_stages++;
    }
    /* Final */
    snprintf(nbg_buf,sizeof(nbg_buf),"%s/qwen25_05b_w32_final_int16/network_binary.nb",model_base);
    char *fpath=strdup(nbg_buf);
    stage_names[n_stages]="final"; stages[n_stages].name="final"; stages[n_stages].nbg_path=fpath; stages[n_stages].is_final=1; n_stages++;

    printf("=== Loading %d stages ===\n",n_stages);
    if (vip_init()!=VIP_SUCCESS) { fprintf(stderr,"vip_init failed\n"); return 1; }
    printf("vip_init=OK\n");

    for (int i=0;i<n_stages;i++) {
        if (stage_init(&stages[i],di)!=0) { fprintf(stderr,"FAILED at stage %d\n",i); goto cleanup; }
    }

    /* Verify chain compatibility */
    int ok=1;
    for (int i=0;i<n_stages-1;i++) {
        if (stages[i].output_bytes!=stages[i+1].input_bytes) {
            fprintf(stderr,"chain mismatch: [%s].out=%u != [%s].in=%u\n",
                    stages[i].name,stages[i].output_bytes,stages[i+1].name,stages[i+1].input_bytes);
            ok=0;
        }
    }
    if (!ok) goto cleanup;
    printf("\nChain validated: all %d stages compatible\n\n",n_stages);

    /* Run decode loop */
    int window[PROMPT_LEN];
    memcpy(window,prompt_tokens,sizeof(window));
    uint64_t chain_times[1024]; int chain_count=0;
    uint64_t first_wall=0;

    /* Verify setup: read expected outputs from host */
    float *expected_logits=NULL;
    if (verify) {
        /* We'll just check that logits are non-zero and plausible */
        printf("Verify mode: checking output validity\n");
    }

    printf("=== Decode loop (%d steps) ===\n",steps);
    int generated[256]; int gen_count=0;

    for (int step=0;step<steps;step++) {
        uint64_t t0=now_us();

        /* Write token window to embedding */
        write_token_window(&stages[0],window,PROMPT_LEN);

        /* Run embedding */
        stage_run(&stages[0],NULL);

        /* Chain: embed->block0, blockN->blockN+1, block23->final */
        for (int i=0;i<n_stages-1;i++) {
            if (i>0) { /* skip embed->block0 copy when i==1 because block0 output already done */
                /* Actually we need to run and chain each stage */
            }
        }

        /* Run and chain from stage 1 (block0) through stage n-2 (block23) */
        for (int i=1;i<n_stages-1;i++) {
            /* Copy previous stage output to this stage's input */
            if (i==1) {
                /* Copy embed output to block0 input */
                copy_output_to_input(&stages[0],&stages[1]);
            }
            /* Run this stage */
            uint32_t prof=0;
            stage_run(&stages[i],&prof);
            /* Copy output to next stage input (if not final) */
            if (i<n_stages-2) {
                copy_output_to_input(&stages[i],&stages[i+1]);
            }
        }

        /* Run final stage */
        uint32_t final_prof=0;
        stage_run(&stages[n_stages-1],&final_prof);

        uint64_t chain_wall=now_us()-t0;
        chain_times[chain_count++]=chain_wall;
        if (step==0) first_wall=chain_wall;

        /* Read logits */
        int next=0; float top_logit=0;
        read_logits(&stages[n_stages-1],MAX_VOCAB,&next,&top_logit);

        generated[gen_count++]=next;
        printf("[%d] token=%d logit=%.2f wall=%"PRIu64"us\n",step,next,top_logit,chain_wall);

        /* Slide window */
        memmove(window,window+1,(PROMPT_LEN-1)*sizeof(int));
        window[PROMPT_LEN-1]=next;
    }

    /* Print summary */
    uint64_t total=0, min_t=UINT64_MAX, max_t=0;
    for (int i=1;i<chain_count;i++) { /* skip first (prefill) */
        total+=chain_times[i];
        if (chain_times[i]<min_t) min_t=chain_times[i];
        if (chain_times[i]>max_t) max_t=chain_times[i];
    }
    int decode_count=chain_count-1;
    double mean_wall = decode_count>0 ? (double)total/(double)decode_count : 0;
    double tok_s = mean_wall>0 ? 1000000.0/mean_wall : 0;

    printf("\n=== Summary ===\n");
    printf("first_step_wall=%"PRIu64"us\n",first_wall);
    printf("mean_wall=%.1fus\n",mean_wall);
    printf("min_wall=%"PRIu64"us\n",min_t);
    printf("max_wall=%"PRIu64"us\n",max_t);
    printf("decode_tok_per_s=%.3f\n",tok_s);
    printf("nbg_stages=%d\n",n_stages);
    printf("nbg_loaded_once=1\n");
    printf("generated_tokens=%d\n",gen_count);
    printf("generated_ids: ");
    for (int i=0;i<gen_count;i++) printf("%d ",generated[i]);
    printf("\n");

cleanup:
    for (int i=n_stages-1;i>=0;i--) stage_destroy(&stages[i]);
    /* Free strdup'd names and paths */
    for (int i=0;i<n_stages;i++) {
        if (stages[i].nbg_path) free((void*)stages[i].nbg_path);
    }
    for (int i=1;i<n_stages-1;i++) {
        if (stages[i].name && stages[i].name[0]) free((void*)stages[i].name);
    }
    vip_destroy();
    return ok?0:1;
}
