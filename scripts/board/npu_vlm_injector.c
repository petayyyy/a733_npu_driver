/*
 * V2d: NPU VLM Embedding Injector
 * Loads SmolVLM model + mmproj, injects NPU embeddings, generates answer.
 * 
 * Compile on Orange Pi:
 *   gcc -O2 npu_vlm_injector.c -o npu_vlm_injector \
 *     -I/home/orangepi/llama.cpp/include -I/home/orangepi/llama.cpp/ggml/include \
 *     -I/home/orangepi/llama.cpp/tools/mtmd \
 *     -L/home/orangepi/llama.cpp/build/bin \
 *     -lllama -lggml -lggml-base -lggml-cpu -lmtmd -lm -lpthread \
 *     -Wl,-rpath,/home/orangepi/llama.cpp/build/bin -lstdc++
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "llama.h"
#include "mtmd.h"

static float * load_embeddings(const char * path, size_t * n_out) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open embeddings: %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    size_t n = sz / sizeof(float);
    float * emb = (float *)malloc(sz);
    if (!emb) { fclose(f); return NULL; }
    fread(emb, 1, sz, f);
    fclose(f);
    *n_out = n;
    fprintf(stderr, "Loaded %zu float32 embeddings (%ld bytes)\n", n, sz);
    return emb;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <model.gguf> <mmproj.gguf> <embeddings.bin> <prompt> [n_gen=128]\n", argv[0]);
        return 1;
    }
    
    const char * model_path  = argv[1];
    const char * mmproj_path = argv[2];
    const char * emb_path    = argv[3];
    const char * prompt      = argv[4];
    int n_gen                = (argc > 5) ? atoi(argv[5]) : 128;
    
    fprintf(stderr, "=== V2d NPU VLM Injector ===\n");
    
    // Load NPU embeddings
    size_t n_emb;
    float * npu_embeddings = load_embeddings(emb_path, &n_emb);
    if (!npu_embeddings) return 1;
    
    // Init llama backend
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
    
    // Load model
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    struct llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Model load failed\n");
        free(npu_embeddings);
        return 1;
    }
    fprintf(stderr, "Model loaded\n");
    
    // Init mtmd with mmproj
    struct mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.n_threads = 2;
    mtmd_params.print_timings = false;
    
    mtmd_context * mctx = mtmd_init_from_file(mmproj_path, model, mtmd_params);
    if (!mctx) {
        fprintf(stderr, "mtmd_init_from_file failed\n");
        llama_model_free(model);
        free(npu_embeddings);
        return 1;
    }
    fprintf(stderr, "Mtmd context created\n");
    
    // Create llama context
    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 1024;
    cparams.n_batch = 1024;
    cparams.n_threads = 2;
    cparams.n_threads_batch = 2;
    
    struct llama_context * ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Context creation failed\n");
        mtmd_free(mctx);
        llama_model_free(model);
        free(npu_embeddings);
        return 1;
    }
    fprintf(stderr, "Context created\n");
    
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    
    // Tokenize the SmolVLM chat prompt with <image> marker
    // Format: <|im_start|>User:\n<image>\n{prompt}<end_of_utterance>\nAssistant:
    char full_prompt[2048];
    snprintf(full_prompt, sizeof(full_prompt),
             "<|im_start|>User:\n<image>\n%s<end_of_utterance>\nAssistant:", prompt);
    
    int all_tokens[1024];
    int n_all = llama_tokenize(vocab, full_prompt, strlen(full_prompt), all_tokens, 1024, true, true);
    if (n_all < 0) {
        fprintf(stderr, "Tokenization failed\n");
        llama_free(ctx);
        mtmd_free(mctx);
        llama_model_free(model);
        free(npu_embeddings);
        return 1;
    }
    fprintf(stderr, "Prompt tokenized: %d tokens\n", n_all);
    
    // Find <image> token
    // In SmolVLM tokenizer, <image> is a special token (ID 49190 or similar)
    // Let's find it by tokenizing just "<image>"
    int img_tokens_arr[8];
    const char * img_marker = "<image>";
    int n_img_tok = llama_tokenize(vocab, img_marker, strlen(img_marker), img_tokens_arr, 8, true, true);
    
    int img_token_id = (n_img_tok > 0) ? img_tokens_arr[0] : -1;
    fprintf(stderr, "Image token ID: %d (from %d tokens for '%s')\n", img_token_id, n_img_tok, img_marker);
    
    // Find the position of image token in the full prompt
    int img_pos = -1;
    for (int i = 0; i < n_all; i++) {
        if (all_tokens[i] == img_token_id) {
            img_pos = i;
            break;
        }
    }
    
    if (img_pos < 0) {
        fprintf(stderr, "Image token not found in prompt. Tokens:");
        for (int i = 0; i < n_all && i < 20; i++) fprintf(stderr, " %d", all_tokens[i]);
        fprintf(stderr, "\n");
        llama_free(ctx);
        mtmd_free(mctx);
        llama_model_free(model);
        free(npu_embeddings);
        return 1;
    }
    fprintf(stderr, "Image token at position %d of %d\n", img_pos, n_all);
    
    // Determine image embedding dimensions
    // SmolVLM: 64 image tokens x 576 dims = 36864 floats
    int n_img_emb_tokens = 64;
    int emb_dim = 576;
    if (n_emb == (size_t)n_img_emb_tokens * emb_dim) {
        fprintf(stderr, "Embeddings: %d tokens x %d dims\n", n_img_emb_tokens, emb_dim);
    } else {
        fprintf(stderr, "WARNING: embeddings size %zu != expected %d\n", n_emb, n_img_emb_tokens * emb_dim);
        // Try to auto-detect dimensions from size
        emb_dim = n_emb / 64;
        if (n_emb % 64 == 0) {
            n_img_emb_tokens = 64;
            fprintf(stderr, "Auto-detected: %d tokens x %d dims\n", n_img_emb_tokens, emb_dim);
        } else {
            n_img_emb_tokens = n_emb / 576;
            if (n_emb % 576 == 0) {
                emb_dim = 576;
                fprintf(stderr, "Auto-detected: %d tokens x %d dims\n", n_img_emb_tokens, emb_dim);
            }
        }
    }
    
    // Build the full sequence:
    // [text before image] [image embeddings (64 tokens)] [text after image]
    int n_before = img_pos;
    int n_after = n_all - img_pos - 1;  // minus the single image token
    int n_total = n_before + n_img_emb_tokens + n_after;
    
    fprintf(stderr, "Sequence: %d text + %d image + %d text = %d total\n",
            n_before, n_img_emb_tokens, n_after, n_total);
    
    // Allocate batch arrays
    llama_pos * pos_arr = (llama_pos *)calloc(n_total, sizeof(llama_pos));
    int32_t * n_seq_arr = (int32_t *)calloc(n_total, sizeof(int32_t));
    llama_seq_id ** seq_arr = (llama_seq_id **)calloc(n_total, sizeof(llama_seq_id *));
    int8_t * logits_arr = (int8_t *)calloc(n_total, sizeof(int8_t));
    llama_token * token_arr = (llama_token *)calloc(n_total, sizeof(llama_token));
    
    for (int i = 0; i < n_total; i++) {
        pos_arr[i] = i;
        n_seq_arr[i] = 1;
        seq_arr[i] = (llama_seq_id *)malloc(sizeof(llama_seq_id));
        seq_arr[i][0] = 0;
        logits_arr[i] = 0;
    }
    
    struct llama_batch batch;
    batch.n_tokens = 0;
    batch.token = NULL;
    batch.embd = NULL;
    batch.pos = pos_arr;
    batch.n_seq_id = n_seq_arr;
    batch.seq_id = seq_arr;
    batch.logits = logits_arr;
    
    // Step 1: Decode text before image
    if (n_before > 0) {
        batch.n_tokens = n_before;
        batch.token = token_arr;
        batch.embd = NULL;
        for (int i = 0; i < n_before; i++) {
            pos_arr[i] = i;
            token_arr[i] = all_tokens[i];
            logits_arr[i] = 0;
        }
        fprintf(stderr, "Decoding %d text tokens before image...\n", n_before);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "Pre-image decode failed\n");
            goto cleanup;
        }
    }
    
    // Step 2: Decode image embeddings
    {
        batch.n_tokens = n_img_emb_tokens;
        batch.token = token_arr;
        batch.embd = npu_embeddings;
        for (int i = 0; i < n_img_emb_tokens; i++) {
            pos_arr[i] = n_before + i;
            logits_arr[i] = 0;
        }
        fprintf(stderr, "Decoding %d image embedding tokens...\n", n_img_emb_tokens);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "Image embed decode failed\n");
            goto cleanup;
        }
    }
    
    // Step 3: Decode text after image
    if (n_after > 0) {
        batch.n_tokens = n_after;
        batch.token = token_arr;
        batch.embd = NULL;
        for (int i = 0; i < n_after; i++) {
            pos_arr[i] = n_before + n_img_emb_tokens + i;
            token_arr[i] = all_tokens[img_pos + 1 + i];
            logits_arr[i] = ((i == n_after - 1) ? 1 : 0);
        }
        fprintf(stderr, "Decoding %d text tokens after image...\n", n_after);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "Post-image decode failed\n");
            goto cleanup;
        }
    }
    
    // Step 4: Generate autoregressively
    fprintf(stderr, "\n=== Answer ===\n");
    
    int bos_id = llama_vocab_bos(vocab);
    int eos_id = llama_vocab_eos(vocab);
    int eot_id = -1;  // <end_of_utterance> token
    
    // Find end_of_utterance token
    const char * eot_str = "<end_of_utterance>";
    int eot_tokens[4];
    int n_eot = llama_tokenize(vocab, eot_str, strlen(eot_str), eot_tokens, 4, true, true);
    if (n_eot > 0) eot_id = eot_tokens[0];
    
    fprintf(stderr, "BOS=%d, EOS=%d, EOT=%d\n", bos_id, eos_id, eot_id);
    
    int prefix_len = n_total;
    int n_gen_out = 0;
    
    // Separate arrays for generation (one token at a time)
    llama_pos gen_pos[1];
    int32_t gen_n_seq[1];
    llama_seq_id * gen_seq[1];
    llama_seq_id gen_seq_val = 0;
    int8_t gen_logits[1];
    llama_token gen_token[1];
    
    gen_seq[0] = &gen_seq_val;
    
    struct llama_batch gen_batch;
    gen_batch.n_tokens = 1;
    gen_batch.token = gen_token;
    gen_batch.embd = NULL;
    gen_batch.pos = gen_pos;
    gen_batch.n_seq_id = gen_n_seq;
    gen_batch.seq_id = gen_seq;
    gen_batch.logits = gen_logits;
    
    // Get logits from the last prefill token
    float * logits_out = llama_get_logits_ith(ctx, n_after > 0 ? (n_after - 1) : (n_before + n_img_emb_tokens - 1));
    if (!logits_out) {
        fprintf(stderr, "Failed to get prefill logits\n");
        goto cleanup;
    }
    
    int n_vocab = llama_vocab_n_tokens(vocab);
    
    for (int i = 0; i < n_gen; i++) {
        // Greedy: find best token
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
        
        // Stop at EOS/EOT
        if ((best_token == eos_id || best_token == eot_id) && i > 4) break;
        
        // Decode and print token
        char piece[256];
        int n_piece = llama_token_to_piece(vocab, best_token, piece, sizeof(piece), 0, true);
        if (n_piece < 0) break;
        fwrite(piece, 1, n_piece, stdout);
        fflush(stdout);
        
        n_gen_out++;
        
        // Decode next token
        gen_pos[0] = prefix_len + i;
        gen_token[0] = best_token;
        gen_n_seq[0] = 1;
        gen_logits[0] = 1;
        
        if (llama_decode(ctx, gen_batch) != 0) break;
        
        // Get logits for next iteration
        logits_out = llama_get_logits_ith(ctx, 0);
        if (!logits_out) break;
    }
    
    fprintf(stderr, "\n\nGenerated %d tokens\n", n_gen_out);
    
cleanup:
    for (int i = 0; i < n_total; i++) free(seq_arr[i]);
    free(pos_arr); free(n_seq_arr); free(seq_arr); free(logits_arr); free(token_arr);
    llama_free(ctx);
    mtmd_free(mctx);
    llama_model_free(model);
    llama_backend_free();
    free(npu_embeddings);
    return 0;
}
