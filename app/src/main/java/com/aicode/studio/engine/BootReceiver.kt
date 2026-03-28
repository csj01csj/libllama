package com.aicode.studio.engine

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log

/**
 * 기기 재부팅 완료(BOOT_COMPLETED) 시 AI 서비스를 자동 재시작한다.
 */
class BootReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Intent.ACTION_BOOT_COMPLETED &&
            intent.action != "android.intent.action.QUICKBOOT_POWERON") return

        Log.d("BootReceiver", "부팅 완료 — 서비스 재시작 시도")

        val mm      = ModelManager(context)
        val modelId = mm.activeModelId
        if (modelId.isBlank()) return

        val model = InferenceConfig.ALL_MODELS.firstOrNull { it.id == modelId }
        if (model == null || !mm.isInstalled(model)) return

        val svcIntent = Intent(context, AIInferenceService::class.java).apply {
            action = AIInferenceService.ACTION_START
            putExtra(InferenceConfig.KEY_MODEL_ID, modelId)
        }
        context.startForegroundService(svcIntent)
        Log.d("BootReceiver", "${model.displayName} 서비스 재시작됨")
    }
}
