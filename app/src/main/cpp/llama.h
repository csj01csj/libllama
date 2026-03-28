/**
 * Minimal llama.cpp C API header for llamatik 0.14.0 / llama.cpp b5000+ (ARM64)
 *
 * Struct layout verified against binary static data at offset 0x93E08 in libllama.so:
 *   n_ctx=512, n_batch=2048, n_ubatch=512 at offsets 0, 4, 8
 *   Total llama_context_params size ≈ 120 bytes (< 256-byte opaque buffer used here)
 *
 * API NOTE (verified by binary analysis of llamatik 0.14.0 libllama.so):
 *   In this build the tokenizer/token functions take const llama_vocab* NOT llama_model*.
 *   llama_model_get_vocab(model) → returns &model->vocab (embedded at offset 0x43C8).
 *   llama_tokenize, llama_token_to_piece, llama_token_is_eog, llama_token_bos/eos
 *   all take const llama_vocab* as their first argument.
 *   Passing llama_model* to these causes SIGSEGV in llama_vocab::impl::tokenize+52
 *   (fault addr 0x170000016e — model struct fields misread as vocab pimpl pointer).
 */
#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Opaque handles ─────────────────────────────────────────────────────────── */
struct llama_model;
struct llama_vocab;    /* new in llamatik 0.14.0 – vocab is now a first-class handle */
struct llama_context;
struct llama_memory_t; /* KV-cache handle returned by llama_get_memory() */
struct llama_sampler;

/* ── Basic types ─────────────────────────────────────────────────────────────── */
typedef int32_t llama_token;
typedef int32_t llama_pos;
typedef int32_t llama_seq_id;

/* ── llama_model_params  (opaque 256-byte buffer; only n_gpu_layers at [0] needed) ─ */
typedef struct {
    int32_t  n_gpu_layers;          /* offset 0  */
    uint8_t  _rest[252];            /* remainder filled by llama_model_default_params() */
} llama_model_params;

/* ── llama_context_params (opaque 256-byte buffer; key fields at known offsets) ─── */
typedef struct {
    uint32_t n_ctx;                 /* offset  0 – context window (tokens) */
    uint32_t n_batch;               /* offset  4 – max tokens in one decode call */
    uint32_t n_ubatch;              /* offset  8 – micro-batch size */
    uint32_t n_seq_max;             /* offset 12 – max parallel sequences */
    int32_t  n_threads;             /* offset 16 – CPU threads for generation */
    int32_t  n_threads_batch;       /* offset 20 – CPU threads for batch processing */
    uint8_t  _rest[232];            /* remainder (enums, floats, callbacks, bools…) */
} llama_context_params;            /*  sizeof = 256 >= actual ≈120, safe oversized buffer */

/* ── llama_batch – exact 56-byte layout (stable across llama.cpp versions) ─── */
typedef struct llama_batch {
    int32_t       n_tokens;         /* offset  0 */
    int32_t       _pad0;            /* offset  4 – alignment padding before pointer */
    llama_token  *token;            /* offset  8 */
    float        *embd;             /* offset 16 */
    llama_pos    *pos;              /* offset 24 */
    int32_t      *n_seq_id;         /* offset 32 */
    llama_seq_id**seq_id;           /* offset 40 */
    int8_t       *logits;           /* offset 48 */
} llama_batch;                      /* sizeof = 56 */

/* ── llama_sampler_chain_params (single bool; returned in register) ────────── */
typedef struct {
    bool no_perf;
} llama_sampler_chain_params;

/* ══════════════════════════════════════════════════════════════════════════════
 *  Public C API
 * ══════════════════════════════════════════════════════════════════════════════ */

/* Backend */
void llama_backend_init(void);
void llama_backend_free(void);

/* Model */
llama_model_params llama_model_default_params(void);
struct llama_model *llama_load_model_from_file(const char *path_model,
                                               llama_model_params params);
void llama_free_model(struct llama_model *model);

/* Vocab – new API: get vocabulary handle embedded in model (returns &model->vocab) */
const struct llama_vocab *llama_model_get_vocab(const struct llama_model *model);
int32_t llama_n_vocab(const struct llama_vocab *vocab);

/* Context */
llama_context_params llama_context_default_params(void);
struct llama_context *llama_new_context_with_model(struct llama_model *model,
                                                    llama_context_params params);
void    llama_free(struct llama_context *ctx);
int32_t llama_n_ctx(const struct llama_context *ctx);
int32_t llama_n_batch(const struct llama_context *ctx);

/* Tokenise – first arg is llama_vocab* (NOT llama_model*) in this binary build */
int32_t llama_tokenize(const struct llama_vocab *vocab,
                       const char *text, int32_t text_len,
                       llama_token *tokens, int32_t n_tokens_max,
                       bool add_special, bool parse_special);

/* Batch */
llama_batch llama_batch_init(int32_t n_tokens_alloc, int32_t embd, int32_t n_seq_max);
void        llama_batch_free(llama_batch batch);

/* Decode */
int llama_decode(struct llama_context *ctx, llama_batch batch);

/* KV-cache / memory management (new API in llamatik 0.14.0) */
struct llama_memory_t *llama_get_memory(struct llama_context *ctx);
void llama_memory_clear(struct llama_memory_t *mem, bool data);
bool llama_memory_seq_rm(struct llama_memory_t *mem, llama_seq_id seq_id,
                          llama_pos p0, llama_pos p1);

/* Logits */
float *llama_get_logits_ith(struct llama_context *ctx, int32_t i);

/* Sampler chain */
llama_sampler_chain_params  llama_sampler_chain_default_params(void);
struct llama_sampler *llama_sampler_chain_init(llama_sampler_chain_params params);
void llama_sampler_chain_add(struct llama_sampler *chain, struct llama_sampler *smpl);
struct llama_sampler *llama_sampler_init_greedy(void);
struct llama_sampler *llama_sampler_init_temp(float t);
struct llama_sampler *llama_sampler_init_top_k(int32_t k);
struct llama_sampler *llama_sampler_init_top_p(float p, size_t min_keep);
struct llama_sampler *llama_sampler_init_dist(uint32_t seed);
struct llama_sampler *llama_sampler_init_penalties(int32_t   penalty_last_n,
                                                    float     penalty_repeat,
                                                    float     penalty_freq,
                                                    float     penalty_present);
llama_token llama_sampler_sample(struct llama_sampler *smpl,
                                  struct llama_context *ctx, int32_t idx);
void llama_sampler_accept(struct llama_sampler *smpl, llama_token token);
void llama_sampler_free(struct llama_sampler *smpl);
void llama_sampler_reset(struct llama_sampler *smpl);

/* Token utilities – first arg is llama_vocab* in this binary build */
int32_t llama_token_to_piece(const struct llama_vocab *vocab, llama_token token,
                              char *buf, int32_t length,
                              int32_t lstrip, bool special);
bool    llama_token_is_eog(const struct llama_vocab *vocab, llama_token token);
llama_token llama_token_bos(const struct llama_vocab *vocab);
llama_token llama_token_eos(const struct llama_vocab *vocab);

#ifdef __cplusplus
}
#endif
