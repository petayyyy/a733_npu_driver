/*
 * V2c: NPU embedding injector USING mtmd API (no llama.cpp patching needed!)
 * Compile: gcc -O2 -o inject_embeds inject_embeds.c -I<inc> -I<ggml> -L<lib> 
 *          -lllama -lggml -lggml-base -lggml-cpu -lmtmd -lm -lpthread -fopenmp 
 *          -Wl,-rpath,<lib> -lstdc++
 *
 * Usage: ./inject_embeds model.gguf mmproj.gguf embeddings.bin image.jpg prompt n_gen
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "llama.h"
#include "mtmd.h"

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "Usage: %s <model.gguf> <mmproj.gguf> <embeddings.bin> <image.jpg> <prompt> [n_gen=64]\n", argv[0]);
        return 1;
    }
    
    const char *model_path  = argv[1];
    const char *mmproj_path = argv[2];
    const char *emb_path    = argv[3];
    const char *image_path  = argv[4];
    const char *prompt      = argv[5];
    int n_gen               = (argc > 6) ? atoi(argv[6]) : 64;
    
    printf("=== V2c NPU Embedding Injector (mtmd API) ===\n");
    
    // Load embeddings
    FILE *ef = fopen(emb_path, "rb");
    if (!ef) { perror("fopen embeddings"); return 1; }
    fseek(ef, 0, SEEK_END);
    long emb_size = ftell(ef);
    fseek(ef, 0, SEEK_SET);
    int n_floats = emb_size / sizeof(float);
    float *embeddings = malloc(emb_size);
    fread(embeddings, 1, emb_size, ef);
    fclose(ef);
    printf("Loaded %d floats (%ld bytes) embeddings\n", n_floats, emb_size);
    
    // Init llama
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
    
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    struct llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "Model load failed\n"); free(embeddings); return 1; }
    
    // Create mtmd context (THIS sets up multimodal handling!)
    struct mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.n_threads = 2;
    mtmd_params.print_timings = false;
    
    printf("Creating mtmd context with mmproj...\n");
    mtmd_context *mctx = mtmd_init_from_file(mmproj_path, model, mtmd_params);
    if (!mctx) {
        fprintf(stderr, "mtmd_init_from_file failed - check mmproj path\n");
        llama_model_free(model);
        free(embeddings);
        return 1;
    }
    printf("mtmd context created OK\n");
    
    // Get the output embeddings buffer from mtmd
    float *mtmd_embd = mtmd_get_output_embd(mctx);
    if (!mtmd_embd) {
        fprintf(stderr, "mtmd_get_output_embd returned NULL\n");
        mtmd_free(mctx);
        llama_model_free(model);
        free(embeddings);
        return 1;
    }
    
    // Copy NPU embeddings into mtmd output buffer
    // The mtmd buffer size should match our embeddings
    memcpy(mtmd_embd, embeddings, emb_size);
    printf("Copied NPU embeddings into mtmd buffer\n");
    
    // Now create llama context with proper n_ctx
    // mtmd context knows the correct token count including image tokens
    // We need to figure out the total token count
    // For SmolVLM: image tokens + prompt tokens + generation
    
    // Actually, we should use the mtmd helper to decode
    // But the helper API is complex. Let's use a simpler path:
    // Use llama-cli's approach: create context, add image, encode, decode
    
    // Create a context that fits everything
    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 512;  // enough for image + prompt + generation
    cparams.n_batch = 512;
    cparams.n_threads = 2;
    cparams.n_threads_batch = 2;
    
    struct llama_context *ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Context creation failed\n");
        mtmd_free(mctx);
        llama_model_free(model);
        free(embeddings);
        return 1;
    }
    
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    
    // Create an image token batch that the mtmd can process
    // Load the image and create image tokens
    struct mtmd_image_tokens *img_tokens = NULL;
    
    // Actually, the mtmd API requires us to load the image through its pipeline
    // This is getting complex. Let's try a different approach:
    // Just use llama_decode directly with the embeddings
    
    // We know: 64 image tokens + prompt text tokens
    // Tokenize the prompt with chat template
    char full_prompt[1024];
    snprintf(full_prompt, sizeof(full_prompt),
             "<|im_start|>User:\n<image>\n%s<end_of_utterance>\nAssistant:", prompt);
    
    int all_tokens[256];
    int n_all = llama_tokenize(vocab, full_prompt, strlen(full_prompt), all_tokens, 256, true, true);
    if (n_all < 0) n_all = 0;
    printf("Prompt tokens: %d\n", n_all);
    
    // Find <image> token
    int img_tok = 49190;
    int img_pos = -1;
    for (int i = 0; i < n_all; i++) {
        if (all_tokens[i] == img_tok) { img_pos = i; break; }
    }
    if (img_pos < 0) { img_pos = 0; }
    
    int n_img = 64;  // 64 image tokens
    int n_before = img_pos;
    int n_after = n_all - img_pos - 1;
    int n_total = n_before + n_img + n_after;
    
    printf("Sequence: %d text + %d img + %d text = %d total\n", n_before, n_img, n_after, n_total);
    
    // Allocate batch arrays
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
    
    // Step 1: Text before image
    if (n_before > 0) {
        struct llama_batch b;
        b.n_tokens = n_before; b.token = token_arr; b.embd = NULL;
        b.pos = pos; b.n_seq_id = n_seq_id_arr; b.seq_id = seq_id_arr; b.logits = logits_arr;
        for (int i = 0; i < n_before; i++) { token_arr[i] = all_tokens[i]; logits_arr[i] = 0; }
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "Pre-img decode fail\n"); goto cleanup; }
    }
    
    // Step 2: Image embeddings
    {
        struct llama_batch b;
        b.n_tokens = n_img; b.token = token_arr; b.embd = embeddings;
        b.pos = pos; b.n_seq_id = n_seq_id_arr; b.seq_id = seq_id_arr; b.logits = logits_arr;
        for (int i = 0; i < n_img; i++) { pos[i] = n_before + i; logits_arr[i] = 0; }
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "Img decode fail\n"); goto cleanup; }
    }
    
    // Step 3: Text after image
    if (n_after > 0) {
        struct llama_batch b;
        b.n_tokens = n_after; b.token = token_arr; b.embd = NULL;
        b.pos = pos; b.n_seq_id = n_seq_id_arr; b.seq_id = seq_id_arr; b.logits = logits_arr;
        for (int i = 0; i < n_after; i++) {
            pos[i] = n_before + n_img + i;
            token_arr[i] = all_tokens[img_pos + 1 + i];
            logits_arr[i] = (i == n_after - 1) ? 1 : 0;
        }
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "Post-img decode fail\n"); goto cleanup; }
    }
    
    // Generate
    printf("\n=== Answer ===\n");
    fflush(stdout);
    
    int bos_id = llama_vocab_bos(vocab);
    int eos_id = llama_vocab_eos(vocab);
    int prefix_len = n_total;
    int n_gen_out = 0;
    
    struct llama_batch gb;
    gb.n_tokens = 1; gb.token = token_arr; gb.embd = NULL;
    gb.pos = pos; gb.n_seq_id = n_seq_id_arr; gb.seq_id = seq_id_arr; gb.logits = logits_arr;
    
    for (int i = 0; i < n_gen; i++) {
        float *logits_out = llama_get_logits_ith(ctx, n_after > 0 ? (n_after - 1 + i) : i);
        if (!logits_out) break;
        
        int best_token = -1;
        float best_val = -INFINITY;
        for (int j = 0; j < 49280; j++) {
            if (j == bos_id) continue;
            if (logits_out[j] > best_val) { best_val = logits_out[j]; best_token = j; }
        }
        
        if (best_token < 0) break;
        if (best_token == eos_id && i > 4) break;
        
        char piece[256];
        int n_piece = llama_token_to_piece(vocab, best_token, piece, sizeof(piece), 0, true);
        if (n_piece < 0) break;
        fwrite(piece, 1, n_piece, stdout);
        fflush(stdout);
        
        pos[0] = prefix_len + i;
        token_arr[0] = best_token;
        logits_arr[0] = 1;
        
        if (llama_decode(ctx, gb) != 0) break;
        n_gen_out++;
    }
    
    printf("\n\nGenerated %d tokens\n", n_gen_out);
    
cleanup:
    for (int i = 0; i < n_total; i++) free(seq_id_arr[i]);
    free(pos); free(n_seq_id_arr); free(seq_id_arr); free(logits_arr); free(token_arr);
    mtmd_free(mctx);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    free(embeddings);
    return 0;
}
    
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
