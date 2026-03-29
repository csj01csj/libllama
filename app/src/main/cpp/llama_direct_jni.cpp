/**
 * Direct llama.cpp JNI bridge for com.aicode.studio.engine.LlamaBridgeDirect.
 *
 * Links directly against the prebuilt libllama.so (from llamatik 0.14.0 AAR) so
 * we have full control over n_ctx / n_batch / n_ubatch and can avoid the SIGABRT
 * crash caused by llamatik's hardcoded 512-maxToken path.
 *
 * Thread-safety contract:
 *   nativeInit / nativeShutdown / nativeGenerate are all called from the SAME
 *   single-thread executor in AIInferenceService (nativeContext dispatcher).
 *   nativeStopGeneration may be called from any thread (uses atomic flag).
 */

#include <jni.h>
#include <android/log.h>
#include <atomic>
#include <cstring>
#include <string>
#include <vector>
#include <setjmp.h>
#include <signal.h>

#include "llama.h"

#define TAG "LlamaDirectJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

/* ── Global engine state ─────────────────────────────────────────────────────── */
static struct llama_model        *g_model           = nullptr;
static const struct llama_vocab  *g_vocab           = nullptr;
static struct llama_context      *g_ctx             = nullptr;
static std::atomic<bool>          g_stop            {false};
static bool                       g_backend_inited  = false;
static bool                       g_vulkan_available = false;

/* ── Vulkan crash-protection state ───────────────────────────────────────────── */
static sigjmp_buf                     g_vk_init_jmp;
static volatile sig_atomic_t          g_in_vk_init = 0;
static struct sigaction               g_saved_sigsegv;
static struct sigaction               g_saved_sigbus;

static void vk_init_crash_handler(int sig, siginfo_t * /*info*/, void * /*ctx*/) {
    if (g_in_vk_init) {
        g_in_vk_init = 0;
        siglongjmp(g_vk_init_jmp, sig);
    }
    /* Not our crash — restore original handler and re-raise */
    sigaction(SIGSEGV, &g_saved_sigsegv, nullptr);
    sigaction(SIGBUS,  &g_saved_sigbus,  nullptr);
    raise(sig);
}

/**
 * Call llama_backend_init() with SIGSEGV/SIGBUS crash protection.
 *
 * On Adreno 830 (Snapdragon 8 Elite / S25+), the Vulkan driver can crash
 * inside llama_backend_init() with SIGSEGV at fault addr 0x63.
 * This wrapper catches the crash via siglongjmp, disables Vulkan via
 * GGML_VK_DISABLE=1, and retries the init CPU-only.
 *
 * Sets g_vulkan_available = true only if the first (Vulkan-enabled) attempt
 * completes without crashing.
 */
static void safe_backend_init() {
    struct sigaction sa{};
    sa.sa_sigaction = vk_init_crash_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &sa, &g_saved_sigsegv);
    sigaction(SIGBUS,  &sa, &g_saved_sigbus);

    g_in_vk_init = 1;
    int crash_sig = sigsetjmp(g_vk_init_jmp, 1);

    if (crash_sig == 0) {
        /* Normal path — Vulkan not disabled, attempt GPU init */
        llama_backend_init();
        g_in_vk_init = 0;
        sigaction(SIGSEGV, &g_saved_sigsegv, nullptr);
        sigaction(SIGBUS,  &g_saved_sigbus,  nullptr);
        g_vulkan_available = true;
        LOGI("llama_backend_init() OK — Vulkan available");
    } else {
        /* Crash caught via signal handler */
        LOGW("llama_backend_init() crashed (signal %d, likely Vulkan driver bug) "
             "— disabling Vulkan and retrying CPU-only", crash_sig);
        sigaction(SIGSEGV, &g_saved_sigsegv, nullptr);
        sigaction(SIGBUS,  &g_saved_sigbus,  nullptr);
        g_in_vk_init = 0;

        setenv("GGML_VK_DISABLE", "1", 1);
        g_vulkan_available = false;

        /* Retry with Vulkan disabled.  CPU backends should initialise cleanly. */
        llama_backend_init();
        LOGI("llama_backend_init() retry (CPU-only) OK");
    }
}

/**
 * Load a model with crash protection for GPU paths.
 *
 * On devices where the Vulkan backend initialises but then crashes during
 * model memory mapping (e.g. Adreno 830 fault addr 0x63 in
 * llama_model_load_from_file), we catch the SIGSEGV, fall back to CPU-only
 * (n_gpu_layers = 0) and retry.
 *
 * @param path         Model file path.
 * @param mparams      Model params (mparams.n_gpu_layers may be mutated to 0
 *                     if the GPU load crashes).
 * @return             Loaded llama_model*, or nullptr on failure.
 */
static struct llama_model* safe_load_model(const char *path,
                                           llama_model_params &mparams)
{
    /* CPU-only path: no need for signal protection */
    if (mparams.n_gpu_layers == 0) {
        return llama_model_load_from_file(path, mparams);
    }

    /* GPU path: install crash guard */
    struct sigaction sa{}, saved_segv{}, saved_bus{};
    sa.sa_sigaction = vk_init_crash_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &sa, &saved_segv);
    sigaction(SIGBUS,  &sa, &saved_bus);

    g_in_vk_init = 1;
    int crash_sig = sigsetjmp(g_vk_init_jmp, 1);

    if (crash_sig == 0) {
        struct llama_model *model = llama_model_load_from_file(path, mparams);
        g_in_vk_init = 0;
        sigaction(SIGSEGV, &saved_segv, nullptr);
        sigaction(SIGBUS,  &saved_bus,  nullptr);
        return model;
    }

    /* Crash during GPU model load — fall back to CPU */
    LOGW("llama_load_model_from_file crashed (signal %d) with n_gpu_layers=%d "
         "— disabling Vulkan and retrying CPU-only", crash_sig, (int)mparams.n_gpu_layers);
    sigaction(SIGSEGV, &saved_segv, nullptr);
    sigaction(SIGBUS,  &saved_bus,  nullptr);
    g_in_vk_init = 0;

    g_vulkan_available = false;
    setenv("GGML_VK_DISABLE", "1", 1);
    mparams.n_gpu_layers = 0;

    return llama_model_load_from_file(path, mparams);
}

/* ── JNI class/method cache ──────────────────────────────────────────────────── */
static JavaVM   *g_jvm           = nullptr;
static jclass    g_cb_class      = nullptr;
static jmethodID g_mid_on_token  = nullptr;
static jmethodID g_mid_on_done   = nullptr;
static jmethodID g_mid_on_error  = nullptr;

/* ─────────────────────────────────────────────────────────────────────────────
 *  JNI_OnLoad – cache the JavaVM
 * ───────────────────────────────────────────────────────────────────────────── */
extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void * /*reserved*/) {
    g_jvm = vm;
    // Do NOT call llama_backend_init() here.
    // JNI_OnLoad fires when System.loadLibrary() is called.
    // llama_backend_init() is called inside nativeInit() where safe_backend_init()
    // handles any Vulkan driver crash via sigsetjmp/siglongjmp.
    return JNI_VERSION_1_6;
}

/* ─────────────────────────────────────────────────────────────────────────────
 *  nativeInit(modelPath, nCtx, nBatch, nUbatch, nGpuLayers, nThreads) : Int
 *  Returns 0 on success, non-zero on error.
 * ───────────────────────────────────────────────────────────────────────────── */
extern "C" JNIEXPORT jint JNICALL
Java_com_aicode_studio_engine_LlamaBridgeDirect_nativeInit(
        JNIEnv *env, jclass /*clazz*/,
        jstring j_model_path,
        jint n_ctx, jint n_batch, jint n_ubatch,
        jint n_gpu_layers, jint n_threads)
{
    /* Initialise ggml backends (signal-protected; auto-falls back to CPU if
     * Vulkan driver crashes).  Guard prevents double-init across model reloads. */
    if (!g_backend_inited) {
        safe_backend_init();
        g_backend_inited = true;
        LOGI("Backend init complete — Vulkan=%s", g_vulkan_available ? "YES" : "NO");
    }

    /* If Vulkan is not available, override GPU layer request to 0 */
    jint effective_gpu_layers = g_vulkan_available ? n_gpu_layers : 0;
    if (effective_gpu_layers == 0 && n_gpu_layers > 0) {
        LOGW("GPU layers requested (%d) but Vulkan unavailable — using CPU only",
             (int)n_gpu_layers);
    }

    /* Tear down any existing session */
    if (g_ctx)   { llama_free(g_ctx);         g_ctx   = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    g_vocab = nullptr;

    const char *model_path = env->GetStringUTFChars(j_model_path, nullptr);
    LOGI("Loading model: %s  ctx=%d batch=%d ubatch=%d gpu_layers=%d (requested=%d) threads=%d",
         model_path, n_ctx, n_batch, n_ubatch, (int)effective_gpu_layers, (int)n_gpu_layers, n_threads);

    /* ── Model params ── */
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = (int32_t)effective_gpu_layers;

    g_model = safe_load_model(model_path, mparams);
    env->ReleaseStringUTFChars(j_model_path, model_path);

    if (!g_model) {
        LOGE("safe_load_model failed");
        return -1;
    }

    /* Update effective GPU status after potential CPU fallback in safe_load_model */
    if (mparams.n_gpu_layers == 0 && effective_gpu_layers > 0) {
        LOGW("Model loaded CPU-only (GPU load crashed) — Vulkan disabled for this session");
    }

    /* Cache vocab handle */
    g_vocab = llama_model_get_vocab(g_model);
    if (!g_vocab) {
        LOGE("llama_model_get_vocab returned null");
        llama_model_free(g_model);
        g_model = nullptr;
        return -1;
    }
    LOGI("Vocab handle: %p  n_vocab=%d", (void*)g_vocab, llama_vocab_n_tokens(g_vocab));

    /* ── Context params ── */
    /*
     * We call llama_context_default_params() to fill our 256-byte buffer with
     * correct defaults (including all enum/bool/float fields we don't touch).
     * Then we override only the fields whose offsets we have verified from the
     * static data at 0x93E08 in libllama.so:
     *   n_ctx    @ offset 0  (uint32_t, default 512)
     *   n_batch  @ offset 4  (uint32_t, default 2048)
     *   n_ubatch @ offset 8  (uint32_t, default 512)
     *   n_threads @ offset 16 (int32_t, default 4)
     *   n_threads_batch @ offset 20 (int32_t, default 4)
     */
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx          = (uint32_t)n_ctx;
    cparams.n_batch        = (uint32_t)n_batch;
    cparams.n_ubatch       = (uint32_t)n_ubatch;
    cparams.n_threads      = (int32_t)n_threads;
    cparams.n_threads_batch = (int32_t)n_threads;

    LOGI("Creating context: n_ctx=%u n_batch=%u n_ubatch=%u n_threads=%d",
         cparams.n_ctx, cparams.n_batch, cparams.n_ubatch, cparams.n_threads);

    g_ctx = llama_init_from_model(g_model, cparams);
    if (!g_ctx) {
        LOGE("llama_init_from_model failed");
        llama_model_free(g_model);
        g_model = nullptr;
        return -2;
    }

    LOGI("Model ready. n_ctx=%d n_batch=%d",
         llama_n_ctx(g_ctx), llama_n_batch(g_ctx));
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 *  nativeIsGpuAvailable() – returns true if Vulkan init succeeded
 * ───────────────────────────────────────────────────────────────────────────── */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_aicode_studio_engine_LlamaBridgeDirect_nativeIsGpuAvailable(
        JNIEnv * /*env*/, jclass /*clazz*/)
{
    return g_vulkan_available ? JNI_TRUE : JNI_FALSE;
}

/* ─────────────────────────────────────────────────────────────────────────────
 *  nativeStopGeneration() – thread-safe, may be called from any thread
 * ───────────────────────────────────────────────────────────────────────────── */
extern "C" JNIEXPORT void JNICALL
Java_com_aicode_studio_engine_LlamaBridgeDirect_nativeStopGeneration(
        JNIEnv * /*env*/, jclass /*clazz*/)
{
    g_stop.store(true);
}

/* ─────────────────────────────────────────────────────────────────────────────
 *  Helper: fire a Java callback safely from the native thread
 * ───────────────────────────────────────────────────────────────────────────── */
static void callback_token(JNIEnv *env, jobject cb, const char *piece) {
    jstring js = env->NewStringUTF(piece);
    env->CallVoidMethod(cb, g_mid_on_token, js);
    env->DeleteLocalRef(js);
}

static void callback_done(JNIEnv *env, jobject cb) {
    env->CallVoidMethod(cb, g_mid_on_done);
}

static void callback_error(JNIEnv *env, jobject cb, const char *msg) {
    jstring js = env->NewStringUTF(msg);
    env->CallVoidMethod(cb, g_mid_on_error, js);
    env->DeleteLocalRef(js);
}

/* ─────────────────────────────────────────────────────────────────────────────
 *  nativeGenerate(prompt, temperature, maxTokens, topK, topP, callback)
 *
 *  Blocks until generation is complete, stopped, or an error occurs.
 *  All token callbacks are delivered synchronously on this thread.
 * ───────────────────────────────────────────────────────────────────────────── */
extern "C" JNIEXPORT void JNICALL
Java_com_aicode_studio_engine_LlamaBridgeDirect_nativeGenerate(
        JNIEnv *env, jclass /*clazz*/,
        jstring j_prompt,
        jfloat temperature, jint max_tokens,
        jint top_k, jfloat top_p,
        jobject callback)
{
    if (!g_model || !g_vocab || !g_ctx) {
        LOGE("nativeGenerate called but model/vocab/context not ready");
        /* Resolve callback methods before using */
        jclass cb_cls = env->GetObjectClass(callback);
        jmethodID mid_err = env->GetMethodID(cb_cls, "onError", "(Ljava/lang/String;)V");
        if (mid_err) {
            jstring js = env->NewStringUTF("모델 미초기화");
            env->CallVoidMethod(callback, mid_err, js);
            env->DeleteLocalRef(js);
        }
        return;
    }

    /* ── Cache callback method IDs ── */
    jclass cb_cls = env->GetObjectClass(callback);
    jmethodID mid_token = env->GetMethodID(cb_cls, "onToken", "(Ljava/lang/String;)V");
    jmethodID mid_done  = env->GetMethodID(cb_cls, "onComplete", "()V");
    jmethodID mid_err   = env->GetMethodID(cb_cls, "onError", "(Ljava/lang/String;)V");
    if (!mid_token || !mid_done || !mid_err) {
        LOGE("Failed to get callback method IDs");
        return;
    }
    g_mid_on_token = mid_token;
    g_mid_on_done  = mid_done;
    g_mid_on_error = mid_err;

    g_stop.store(false);

    /* ── Clear KV cache before each new generation ────────────────────────────
     * Without this, the second message fails: context still holds position data
     * from the previous run, so llama_decode for the new prompt returns non-zero.
     * llama_memory_seq_rm(mem, seq_id=0, p0=-1, p1=-1) removes all tokens in
     * sequence 0 (the sentinel values -1/-1 mean "entire sequence"). */
    {
        llama_memory_t mem = llama_get_memory(g_ctx);
        if (mem) {
            llama_memory_seq_rm(mem, 0, -1, -1);
            LOGI("KV cache cleared for new generation");
        }
    }

    /* ── Tokenise prompt ── */
    const char *prompt_cstr = env->GetStringUTFChars(j_prompt, nullptr);
    const int n_prompt_chars = (int)strlen(prompt_cstr);

    const int n_ctx_size = llama_n_ctx(g_ctx);
    std::vector<llama_token> prompt_tokens(n_ctx_size);
    int n_prompt_tokens = llama_tokenize(
            g_vocab, prompt_cstr, n_prompt_chars,
            prompt_tokens.data(), n_ctx_size,
            true,   /* add_special */
            true);  /* parse_special – needed for <|im_start|> etc. */
    env->ReleaseStringUTFChars(j_prompt, prompt_cstr);

    if (n_prompt_tokens < 0) {
        LOGE("Tokenisation failed (returned %d)", n_prompt_tokens);
        callback_error(env, callback, "토크나이즈 실패");
        return;
    }
    if (n_prompt_tokens >= n_ctx_size) {
        LOGE("Prompt too long: %d >= %d", n_prompt_tokens, n_ctx_size);
        callback_error(env, callback, "프롬프트가 너무 깁니다");
        return;
    }
    prompt_tokens.resize(n_prompt_tokens);
    LOGI("Prompt tokenised: %d tokens (ctx=%d)", n_prompt_tokens, n_ctx_size);

    /* ── Prefill ── */
    {
        llama_batch batch = llama_batch_init(n_prompt_tokens, 0, 1);
        batch.n_tokens = n_prompt_tokens;
        for (int i = 0; i < n_prompt_tokens; ++i) {
            batch.token[i]     = prompt_tokens[i];
            batch.pos[i]       = (llama_pos)i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = (i == n_prompt_tokens - 1) ? 1 : 0;
        }
        int ret = llama_decode(g_ctx, batch);
        llama_batch_free(batch);
        if (ret != 0) {
            LOGE("llama_decode (prefill) failed: %d", ret);
            callback_error(env, callback, "프리필 실패");
            return;
        }
    }

    /* ── Build sampler chain ── */
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler *sampler = llama_sampler_chain_init(sparams);

    if (top_k > 0)    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k));
    if (top_p < 1.0f) llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
    if (temperature > 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(0xDEADBEEF));
    } else {
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    }

    /* ── Generation loop ── */
    int n_past = n_prompt_tokens;
    int n_generated = 0;
    char piece_buf[256];
    bool stopped_by_eog = false;

    /* UTF-8 accumulation buffer.
     * llama_token_to_piece may return the first byte(s) of a multi-byte codepoint
     * (e.g. 0xED for a Korean char) before the continuation bytes arrive in the
     * next token.  Passing an incomplete sequence to NewStringUTF causes a
     * JNI abort ("illegal continuation byte 0").  We buffer raw bytes here and
     * only flush complete UTF-8 codepoints to Java.  */
    std::string utf8_pending;

    /* Returns the byte-length of complete UTF-8 codepoints at the start of s. */
    auto complete_utf8_len = [](const std::string &s) -> size_t {
        size_t i = 0;
        while (i < s.size()) {
            unsigned char c = (unsigned char)s[i];
            size_t char_len;
            if      (c < 0x80) char_len = 1;
            else if (c < 0xC0) { ++i; continue; }  /* stray continuation – skip */
            else if (c < 0xE0) char_len = 2;
            else if (c < 0xF0) char_len = 3;
            else                char_len = 4;
            if (i + char_len > s.size()) break;     /* incomplete – stop here */
            i += char_len;
        }
        return i;
    };

    while (n_generated < (int)max_tokens && n_past < n_ctx_size) {
        if (g_stop.load()) {
            LOGI("Generation stopped by request");
            break;
        }

        /* Sample next token */
        llama_token new_token = llama_sampler_sample(sampler, g_ctx, -1);
        llama_sampler_accept(sampler, new_token);

        if (llama_vocab_is_eog(g_vocab, new_token)) {
            stopped_by_eog = true;
            break;
        }

        /* Convert token → UTF-8 piece and buffer it */
        int piece_len = llama_token_to_piece(
                g_vocab, new_token,
                piece_buf, (int)sizeof(piece_buf) - 1,
                0,      /* lstrip */
                false); /* special */
        if (piece_len > 0) {
            utf8_pending.append(piece_buf, piece_len);

            /* Flush only complete codepoints so NewStringUTF never sees
             * a truncated multi-byte sequence (avoids JNI abort on Korean etc.) */
            size_t complete = complete_utf8_len(utf8_pending);
            if (complete > 0) {
                std::string to_send(utf8_pending.data(), complete);
                utf8_pending.erase(0, complete);
                jstring js = env->NewStringUTF(to_send.c_str());
                if (js) {
                    env->CallVoidMethod(callback, g_mid_on_token, js);
                    env->DeleteLocalRef(js);
                }
            }
        }

        /* Decode for next step */
        llama_batch step_batch = llama_batch_init(1, 0, 1);
        step_batch.n_tokens     = 1;
        step_batch.token[0]     = new_token;
        step_batch.pos[0]       = (llama_pos)n_past;
        step_batch.n_seq_id[0]  = 1;
        step_batch.seq_id[0][0] = 0;
        step_batch.logits[0]    = 1;

        int ret = llama_decode(g_ctx, step_batch);
        llama_batch_free(step_batch);
        if (ret != 0) {
            LOGE("llama_decode (step %d) failed: %d", n_generated, ret);
            llama_sampler_free(sampler);
            callback_error(env, callback, "생성 중 디코드 실패");
            return;
        }

        ++n_past;
        ++n_generated;
    }

    /* Flush any remaining bytes (e.g. a trailing incomplete sequence → drop silently) */
    if (!utf8_pending.empty()) {
        size_t complete = complete_utf8_len(utf8_pending);
        if (complete > 0) {
            std::string to_send(utf8_pending.data(), complete);
            jstring js = env->NewStringUTF(to_send.c_str());
            if (js) {
                env->CallVoidMethod(callback, g_mid_on_token, js);
                env->DeleteLocalRef(js);
            }
        }
        utf8_pending.clear();
    }

    llama_sampler_free(sampler);
    LOGI("Generation complete: %d tokens (eog=%d)", n_generated, (int)stopped_by_eog);
    callback_done(env, callback);
}

/* ─────────────────────────────────────────────────────────────────────────────
 *  nativeShutdown – free model + context
 * ───────────────────────────────────────────────────────────────────────────── */
extern "C" JNIEXPORT void JNICALL
Java_com_aicode_studio_engine_LlamaBridgeDirect_nativeShutdown(
        JNIEnv * /*env*/, jclass /*clazz*/)
{
    g_stop.store(true);
    if (g_ctx)   { llama_free(g_ctx);         g_ctx   = nullptr; }
    g_vocab = nullptr;  /* vocab is embedded in model; freed with it */
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    if (g_backend_inited) {
        llama_backend_free();
        g_backend_inited = false;
    }
    LOGI("Shutdown complete");
}
