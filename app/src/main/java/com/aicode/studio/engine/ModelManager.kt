package com.aicode.studio.engine

import android.app.DownloadManager
import android.content.Context
import android.content.SharedPreferences
import android.net.Uri
import android.util.Log
import java.io.File

/**
 * GGUF 모델 파일 다운로드 / 삭제 / 영속 관리.
 * 저장 경로: getExternalFilesDir("models")/<fileName>
 * 다운로드는 Android DownloadManager 사용 (백그라운드, 재시도, 알림 자동 처리)
 */
class ModelManager(private val context: Context) {

    companion object {
        private const val TAG   = "ModelManager"
        private const val PREFS = "model_prefs"
    }

    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
    private val dm: DownloadManager =
        context.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager

    fun modelsDir(): File =
        (context.getExternalFilesDir("models") ?: context.filesDir.resolve("models"))
            .also { it.mkdirs() }

    fun modelFile(model: InferenceConfig.ModelDef): File = File(modelsDir(), model.fileName)

    fun isInstalled(model: InferenceConfig.ModelDef): Boolean =
        prefs.getBoolean("installed_${model.id}", false) &&
        modelFile(model).exists() && modelFile(model).length() > 0L

    fun markInstalled(model: InferenceConfig.ModelDef) {
        prefs.edit().putBoolean("installed_${model.id}", true).apply()
    }

    var activeModelId: String
        get()  = prefs.getString("active_model", "") ?: ""
        set(v) = prefs.edit().putString("active_model", v).apply()

    fun activeModel(): InferenceConfig.ModelDef? =
        InferenceConfig.ALL_MODELS.firstOrNull { it.id == activeModelId }

    // ─── DownloadManager ──────────────────────────────────────────────────────

    data class DlInfo(val status: Int, val downloaded: Long, val total: Long) {
        val pct      get() = if (total > 0) downloaded.toFloat() / total else 0f
        val isActive get() = status == DownloadManager.STATUS_RUNNING
                          || status == DownloadManager.STATUS_PENDING
                          || status == DownloadManager.STATUS_PAUSED
        val isDone   get() = status == DownloadManager.STATUS_SUCCESSFUL
        val isFailed get() = status == DownloadManager.STATUS_FAILED
    }

    /** DownloadManager에 다운로드 등록 후 ID 반환. 앱이 꺼져도 백그라운드로 계속 진행됨. */
    fun startSystemDownload(model: InferenceConfig.ModelDef): Long {
        // 기존 다운로드가 있으면 제거
        val oldId = prefs.getLong("dl_id_${model.id}", -1L)
        if (oldId != -1L) runCatching { dm.remove(oldId) }

        val req = DownloadManager.Request(Uri.parse(model.downloadUrl))
            .setTitle(model.displayName)
            .setDescription("AI 모델 다운로드 중...")
            .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
            .setDestinationUri(Uri.fromFile(modelFile(model)))
            .setAllowedOverMetered(true)
            .setAllowedOverRoaming(false)

        val id = dm.enqueue(req)
        prefs.edit().putLong("dl_id_${model.id}", id).apply()
        Log.d(TAG, "DownloadManager enqueued id=$id for ${model.id}")
        return id
    }

    /** 현재 다운로드 상태 조회. null이면 등록된 다운로드 없음. */
    fun queryDownload(model: InferenceConfig.ModelDef): DlInfo? {
        val id = prefs.getLong("dl_id_${model.id}", -1L)
        if (id == -1L) return null
        dm.query(DownloadManager.Query().setFilterById(id)).use { c ->
            if (!c.moveToFirst()) return null
            return DlInfo(
                status     = c.getInt(c.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS)),
                downloaded = c.getLong(c.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR)),
                total      = c.getLong(c.getColumnIndexOrThrow(DownloadManager.COLUMN_TOTAL_SIZE_BYTES))
            )
        }
    }

    fun cancelDownload(modelId: String) {
        val id = prefs.getLong("dl_id_${modelId}", -1L)
        if (id != -1L) {
            runCatching { dm.remove(id) }
            prefs.edit().remove("dl_id_${modelId}").apply()
        }
        InferenceConfig.ALL_MODELS.firstOrNull { it.id == modelId }?.let {
            modelFile(it).delete()
        }
    }

    fun deleteModel(model: InferenceConfig.ModelDef): Boolean {
        val ok = modelFile(model).delete()
        if (ok) prefs.edit().remove("installed_${model.id}").apply()
        if (activeModelId == model.id) activeModelId = ""
        return ok
    }
}
