/*
 * V2c: Minimal NPU embedding injector for SmolVLM via llama.cpp C API
 * Compile: gcc -O2 -o inject_embeds inject_embeds.c -I<inc> -I<ggml> -L<lib> -lllama -lggml -lggml-base -lggml-cpu -lm -lpthread -fopenmp -Wl,-rpath,<lib>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "llama.h"

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <model.gguf> <embeddings.bin> <prompt> <n_img_tokens> <embd_dim>\n", argv[0]);
        return 1;
    }
    
    const char *model_path = argv[1];
    const char *emb_path   = argv[2];
    const char *prompt     = argv[3];
    int n_img_tokens       = atoi(argv[4]);
    int embd_dim           = atoi(argv[5]);
    int n_gen              = (argc > 6) ? atoi(argv[6]) : 128;
    
    printf("=== V2c NPU Embedding Injector ===\n");
    
    // Load embeddings
    FILE *ef = fopen(emb_path, "rb");
    if (!ef) { perror("fopen embeddings"); return 1; }
    int n_floats = n_img_tokens * embd_dim;
    float *embeddings = malloc(n_floats * sizeof(float));
    if (fread(embeddings, sizeof(float), n_floats, ef) != (size_t)n_floats) {
        fprintf(stderr, "Embeddings file too small\n"); fclose(ef); free(embeddings); return 1;
    }
    fclose(ef);
    printf("Loaded %d x %d embeddings\n", n_img_tokens, embd_dim);
    
    // Init
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
    
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    
    struct llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); free(embeddings); return 1; }
    
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);
    int n_embd = llama_model_n_embd(model);
    printf("Model: vocab=%d embd=%d\n", n_vocab, n_embd);
    
    if (n_embd != embd_dim) {
        fprintf(stderr, "Embedding dim mismatch: model=%d file=%d\n", n_embd, embd_dim);
        llama_model_free(model); free(embeddings); return 1;
    }
    
    // Context
    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_img_tokens + 256 + n_gen;
    cparams.n_batch = n_img_tokens + 256;
    cparams.n_threads = 2;
    cparams.n_threads_batch = 2;
    
    struct llama_context *ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); llama_model_free(model); free(embeddings); return 1; }
    
    // Tokenize prompt
    int text_tokens[512];
    int n_text = llama_tokenize(vocab, prompt, strlen(prompt), text_tokens, 512, true, true);
    if (n_text < 0) n_text = 0;
    printf("Text tokens: %d\n", n_text);
    
    int n_total = n_img_tokens + n_text;
    
    // Build batch for image embeddings
    struct llama_batch img_batch;
    img_batch.n_tokens = n_img_tokens;
    
    llama_pos *pos = calloc(n_total, sizeof(llama_pos));
    int32_t *n_seq_id_arr = calloc(n_total, sizeof(int32_t));
    llama_seq_id **seq_id_arr = calloc(n_total, sizeof(llama_seq_id*));
    int8_t *logits_arr = calloc(n_total, sizeof(int8_t));
    llama_token *token_arr = calloc(n_total, sizeof(llama_token));
    
    for (int i = 0; i < n_total; i++) {
        pos[i] = i;
        n_seq_id_arr[i] = 1;
        seq_id_arr[i] = malloc(sizeof(llama_seq_id));
        seq_id_arr[i][0] = 0;
    }
    
    // Image batch - embedding injection
    img_batch.token = token_arr;
    img_batch.embd = embeddings;  // THE KEY: use embeddings instead of tokens
    img_batch.pos = pos;
    img_batch.n_seq_id = n_seq_id_arr;
    img_batch.seq_id = seq_id_arr;
    img_batch.logits = logits_arr;
    
    printf("Decoding %d image embeddings...\n", n_img_tokens);
    if (llama_decode(ctx, img_batch) != 0) {
        fprintf(stderr, "Image decode failed\n");
        goto cleanup;
    }
    
    // Text batch
    printf("Decoding %d text tokens...\n", n_text);
    struct llama_batch text_batch;
    text_batch.n_tokens = n_text;
    text_batch.token = token_arr;
    text_batch.embd = NULL;  // NULL = use token IDs
    text_batch.pos = pos;
    text_batch.n_seq_id = n_seq_id_arr;
    text_batch.seq_id = seq_id_arr;
    text_batch.logits = logits_arr;
    
    for (int i = 0; i < n_text; i++) {
        pos[i] = n_img_tokens + i;
        token_arr[i] = text_tokens[i];
        logits_arr[i] = (i == n_text - 1) ? 1 : 0;
    }
    
    if (llama_decode(ctx, text_batch) != 0) {
        fprintf(stderr, "Text decode failed\n");
        goto cleanup;
    }
    
    // Generate
    printf("\n=== Answer ===\n");
    fflush(stdout);
    
    int bos_id = llama_vocab_bos(vocab);
    int eos_id = llama_vocab_eos(vocab);
    int n_gen_out = 0;
    
    struct llama_batch gen_batch;
    gen_batch.n_tokens = 1;
    gen_batch.token = token_arr;
    gen_batch.embd = NULL;
    gen_batch.pos = pos;
    gen_batch.n_seq_id = n_seq_id_arr;
    gen_batch.seq_id = seq_id_arr;
    gen_batch.logits = logits_arr;
    
    for (int i = 0; i < n_gen; i++) {
        float *logits_out = llama_get_logits_ith(ctx, n_text - 1 + i);
        if (!logits_out) break;
        
        int best_token = -1;
        float best_val = -INFINITY;
        for (int j = 0; j < n_vocab; j++) {
            if (j == bos_id) continue;
            if (logits_out[j] > best_val) {
                best_val = logits_out[j];
                best_token = j;
            }
        }
        
        if (best_token == eos_id || best_token < 0) break;
        
        char piece[256];
        int n_piece = llama_token_to_piece(vocab, best_token, piece, sizeof(piece), 0, true);
        if (n_piece < 0) break;
        fwrite(piece, 1, n_piece, stdout);
        fflush(stdout);
        
        pos[0] = n_img_tokens + n_text + i;
        token_arr[0] = best_token;
        logits_arr[0] = 1;
        
        if (llama_decode(ctx, gen_batch) != 0) break;
        n_gen_out++;
    }
    
    printf("\n\nGenerated %d tokens\n", n_gen_out);
    
cleanup:
    for (int i = 0; i < n_total; i++) free(seq_id_arr[i]);
    free(pos); free(n_seq_id_arr); free(seq_id_arr); free(logits_arr); free(token_arr);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    free(embeddings);
    return 0;
}
