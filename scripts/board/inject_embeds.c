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
    
    // Don't use <image> token at all - just prepend embeddings
    // Tokenize ONLY the text prompt (no image token)
    int all_tokens[512];
    int n_all = llama_tokenize(vocab, prompt, strlen(prompt), all_tokens, 512, true, true);
    if (n_all < 0) n_all = 0;
    printf("Text prompt tokens: %d\n", n_all);
    
    // Place all image embeddings at the start, then text tokens after
    int n_before = 0;
    int n_after = n_all;
    int image_pos = 0;
    int n_total = n_before + n_img_tokens + n_after;
    printf("Sequence: %d image + %d text = %d total\n", n_img_tokens, n_after, n_total);
    
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
    
    // Step 1: Decode text before image
    if (n_before > 0) {
        struct llama_batch before_batch;
        before_batch.n_tokens = n_before;
        before_batch.token = token_arr;
        before_batch.embd = NULL;
        before_batch.pos = pos;
        before_batch.n_seq_id = n_seq_id_arr;
        before_batch.seq_id = seq_id_arr;
        before_batch.logits = logits_arr;
        for (int i = 0; i < n_before; i++) {
            pos[i] = i;
            token_arr[i] = all_tokens[i];
            logits_arr[i] = 0;
        }
        printf("Decoding %d text tokens before image...\n", n_before);
        if (llama_decode(ctx, before_batch) != 0) {
            fprintf(stderr, "Pre-image decode failed\n");
            goto cleanup;
        }
    }
    
    // Step 2: Decode image embeddings
    {
        struct llama_batch img_batch;
        img_batch.n_tokens = n_img_tokens;
        img_batch.token = token_arr;
        img_batch.embd = embeddings;
        img_batch.pos = pos;
        img_batch.n_seq_id = n_seq_id_arr;
        img_batch.seq_id = seq_id_arr;
        img_batch.logits = logits_arr;
        for (int i = 0; i < n_img_tokens; i++) {
            pos[i] = n_before + i;
            logits_arr[i] = 0;
        }
        printf("Decoding %d image embeddings...\n", n_img_tokens);
        if (llama_decode(ctx, img_batch) != 0) {
            fprintf(stderr, "Image decode failed\n");
            goto cleanup;
        }
    }
    
    // Step 3: Decode text after image
    if (n_after > 0) {
        struct llama_batch after_batch;
        after_batch.n_tokens = n_after;
        after_batch.token = token_arr;
        after_batch.embd = NULL;
        after_batch.pos = pos;
        after_batch.n_seq_id = n_seq_id_arr;
        after_batch.seq_id = seq_id_arr;
        after_batch.logits = logits_arr;
        for (int i = 0; i < n_after; i++) {
            pos[i] = n_before + n_img_tokens + i;
            token_arr[i] = all_tokens[i];  // just use all tokens (no image placeholder in prompt)
            logits_arr[i] = (i == n_after - 1) ? 1 : 0;
        }
        printf("Decoding %d text tokens after image...\n", n_after);
        if (llama_decode(ctx, after_batch) != 0) {
            fprintf(stderr, "Post-image decode failed\n");
            goto cleanup;
        }
    }
    
    // Generate
    printf("\n=== Answer ===\n");
    fflush(stdout);
    
    int bos_id = llama_vocab_bos(vocab);
    int eos_id = llama_vocab_eos(vocab);
    printf("BOS=%d, EOS=%d\n", bos_id, eos_id);
    int n_gen_out = 0;
    int prefix_len = n_before + n_img_tokens + n_after;
    
    struct llama_batch gen_batch;
    gen_batch.n_tokens = 1;
    gen_batch.token = token_arr;
    gen_batch.embd = NULL;
    gen_batch.pos = pos;
    gen_batch.n_seq_id = n_seq_id_arr;
    gen_batch.seq_id = seq_id_arr;
    gen_batch.logits = logits_arr;
    
    for (int i = 0; i < n_gen; i++) {
        float *logits_out = llama_get_logits_ith(ctx, n_after > 0 ? (n_after - 1 + i) : i);
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
        
        if (best_token < 0) break;
        // Don't stop at EOS for this test - let it generate full response
        if (best_token == eos_id && i > 8) break;  // only stop EOS after several tokens
        
        // Debug: show top-5 tokens
        if (i < 3) {
            printf("[top5:");
            // Simple bubble to find top 5
            int top_idx[5] = {-1,-1,-1,-1,-1};
            float top_val[5] = {-INFINITY,-INFINITY,-INFINITY,-INFINITY,-INFINITY};
            for (int j = 0; j < n_vocab && j < 49280; j++) {
                if (j == bos_id) continue;
                float v = logits_out[j];
                for (int k = 0; k < 5; k++) {
                    if (v > top_val[k]) {
                        for (int l = 4; l > k; l--) { top_idx[l] = top_idx[l-1]; top_val[l] = top_val[l-1]; }
                        top_idx[k] = j; top_val[k] = v; break;
                    }
                }
            }
            for (int k = 0; k < 5; k++) {
                if (top_idx[k] >= 0) {
                    char p[64]; llama_token_to_piece(vocab, top_idx[k], p, sizeof(p), 0, true);
                    printf("%d:'%s'=%.1f ", top_idx[k], p, top_val[k]);
                }
            }
            printf("] ");
        }
        
        char piece[256];
        int n_piece = llama_token_to_piece(vocab, best_token, piece, sizeof(piece), 0, true);
        if (n_piece < 0) break;
        fwrite(piece, 1, n_piece, stdout);
        fflush(stdout);
        
        pos[0] = prefix_len + i;
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
