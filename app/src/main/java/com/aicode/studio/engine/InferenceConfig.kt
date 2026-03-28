package com.aicode.studio.engine

/**
 * 지원 모델 + IPC 상수.
 * 모델 파일: /sdcard/Android/data/com.aicode.studio/files/models/<fileName>
 * 형식: GGUF Q4_K_M (4비트 양자화)
 * 엔진: llamatik (llama.cpp Android 바인딩)
 */
object InferenceConfig {

    // ── IPC ─────────────────────────────────────────────────
    const val ACTION_BIND_AI = "com.aicode.studio.engine.BIND_AI"

    const val MSG_SEND_PROMPT  = 1
    const val MSG_TOKEN_STREAM = 2
    const val MSG_GEN_COMPLETE = 3
    const val MSG_GET_STATUS   = 4
    const val MSG_STATUS_REPLY = 5
    const val MSG_STOP_GEN     = 6
    const val MSG_SET_CONTEXT   = 7
    const val MSG_SET_THINKING  = 8
    const val MSG_CLEAR_HISTORY = 9
    const val MSG_SET_HISTORY   = 10

    const val KEY_PROMPT       = "prompt"
    const val KEY_TOKEN        = "token"
    const val KEY_MODEL_ID     = "model_id"
    const val KEY_CONTEXT_JSON = "context_json"
    const val KEY_THINKING     = "thinking_enabled"
    const val KEY_HISTORY      = "history"

    // ── 알림 ────────────────────────────────────────────────
    const val NOTIF_CHANNEL_ID = "ai_inference"
    const val NOTIF_ID         = 1001

    // ── HuggingFace 다운로드 베이스 URL ──────────────────────
    private const val HF = "https://huggingface.co"

    // ── 모델 정의 ────────────────────────────────────────────
    data class ModelDef(
        val id              : String,
        val displayName     : String,
        val series          : String,
        val paramsBillion   : Float,
        val downloadSizeGb  : Float,
        val minRamGb        : Int,
        val supportsThinking: Boolean,
        val downloadUrl     : String,
        val fileName        : String
    )

    val ALL_MODELS = listOf(

        // ── Qwen2.5-Coder 시리즈 (코딩 특화, thinking 미지원) ──
        ModelDef(
            id               = "qwen25_coder_05b",
            displayName      = "Qwen2.5-Coder 0.5B",
            series           = "Qwen2.5-Coder",
            paramsBillion    = 0.5f,
            downloadSizeGb   = 0.4f,
            minRamGb         = 2,
            supportsThinking = false,
            downloadUrl      = "$HF/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
            fileName         = "qwen25_coder_0.5b_q4_k_m.gguf"
        ),
        ModelDef(
            id               = "qwen25_coder_15b",
            displayName      = "Qwen2.5-Coder 1.5B",
            series           = "Qwen2.5-Coder",
            paramsBillion    = 1.5f,
            downloadSizeGb   = 1.1f,
            minRamGb         = 3,
            supportsThinking = false,
            downloadUrl      = "$HF/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
            fileName         = "qwen25_coder_1.5b_q4_k_m.gguf"
        ),
        ModelDef(
            id               = "qwen25_coder_3b",
            displayName      = "Qwen2.5-Coder 3B",
            series           = "Qwen2.5-Coder",
            paramsBillion    = 3.0f,
            downloadSizeGb   = 2.0f,
            minRamGb         = 4,
            supportsThinking = false,
            downloadUrl      = "$HF/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/qwen2.5-coder-3b-instruct-q4_k_m.gguf",
            fileName         = "qwen25_coder_3b_q4_k_m.gguf"
        ),
        ModelDef(
            id               = "qwen25_coder_7b",
            displayName      = "Qwen2.5-Coder 7B",
            series           = "Qwen2.5-Coder",
            paramsBillion    = 7.0f,
            downloadSizeGb   = 4.7f,
            minRamGb         = 5,
            supportsThinking = false,
            downloadUrl      = "$HF/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
            fileName         = "qwen25_coder_7b_q4_k_m.gguf"
        ),

        // ── Qwen3 시리즈 (thinking/non-thinking 전환 가능) ──────
        ModelDef(
            id               = "qwen3_06b",
            displayName      = "Qwen3 0.6B",
            series           = "Qwen3",
            paramsBillion    = 0.6f,
            downloadSizeGb   = 0.4f,
            minRamGb         = 2,
            supportsThinking = true,
            downloadUrl      = "$HF/bartowski/Qwen_Qwen3-0.6B-GGUF/resolve/main/Qwen_Qwen3-0.6B-Q4_K_M.gguf",
            fileName         = "qwen3_0.6b_q4_k_m.gguf"
        ),
        ModelDef(
            id               = "qwen3_17b",
            displayName      = "Qwen3 1.7B",
            series           = "Qwen3",
            paramsBillion    = 1.7f,
            downloadSizeGb   = 1.1f,
            minRamGb         = 3,
            supportsThinking = true,
            downloadUrl      = "$HF/bartowski/Qwen_Qwen3-1.7B-GGUF/resolve/main/Qwen_Qwen3-1.7B-Q4_K_M.gguf",
            fileName         = "qwen3_1.7b_q4_k_m.gguf"
        ),
        ModelDef(
            id               = "qwen3_4b",
            displayName      = "Qwen3 4B",
            series           = "Qwen3",
            paramsBillion    = 4.0f,
            downloadSizeGb   = 2.5f,
            minRamGb         = 4,
            supportsThinking = true,
            downloadUrl      = "$HF/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf",
            fileName         = "qwen3_4b_q4_k_m.gguf"
        )
    )
}
