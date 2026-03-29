package com.aicode.studio.engine

/**
 * Direct llama.cpp JNI bridge (replaces llamatik's LlamaBridge).
 *
 * Provides full control over n_ctx / n_batch / n_ubatch / n_gpu_layers,
 * eliminating the SIGABRT crash caused by llamatik's restricted API.
 *
 * All public methods MUST be called from the same single-thread executor
 * (nativeContext dispatcher in AIInferenceService), EXCEPT stopGeneration()
 * which is thread-safe.
 */
object LlamaBridgeDirect {

    init {
        // Load in dependency order: transitive deps → llama → our JNI bridge.
        // ggml-vulkan must be loaded BEFORE llama so the backend can auto-register.
        // Android 6+ auto-loads DT_NEEDED, but explicit ordering avoids edge cases.
        runCatching { System.loadLibrary("omp") }
        System.loadLibrary("ggml-base")
        System.loadLibrary("ggml-cpu")
        runCatching { System.loadLibrary("ggml-vulkan") }  // optional; crash-safe via JNI signal handler
        System.loadLibrary("ggml")
        System.loadLibrary("llama")
        System.loadLibrary("llama_direct")
    }

    interface TokenCallback {
        fun onToken(piece: String)
        fun onComplete()
        fun onError(error: String)
    }

    /**
     * Load model and create inference context.
     * @return 0 on success, non-zero on error.
     */
    external fun nativeInit(
        modelPath: String,
        nCtx: Int,
        nBatch: Int,
        nUbatch: Int,
        nGpuLayers: Int,
        nThreads: Int
    ): Int

    /**
     * Run generation synchronously. Calls [callback] for each token piece.
     * Blocks until complete, stopped, or an error occurs.
     */
    external fun nativeGenerate(
        prompt: String,
        temperature: Float,
        maxTokens: Int,
        topK: Int,
        topP: Float,
        callback: TokenCallback
    )

    /** Thread-safe: signals the running generation to stop. */
    external fun nativeStopGeneration()

    /** Free context and model. Must be called from the native executor thread. */
    external fun nativeShutdown()

    /**
     * Returns true if Vulkan backend initialised successfully.
     * Only meaningful after [nativeInit] has been called at least once.
     * Returns false if Vulkan crashed during init (auto-fallback to CPU).
     */
    external fun nativeIsGpuAvailable(): Boolean
}
