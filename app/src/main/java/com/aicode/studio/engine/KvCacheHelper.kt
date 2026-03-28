package com.aicode.studio.engine

import android.util.Log
import org.json.JSONObject
import java.io.File

object KvCacheHelper {

    private const val TAG = "KvCacheHelper"

    fun optimizeForDevice(modelFile: File, profile: HardwareAnalyzer.HardwareProfile) {
        runCatching {
            val configFile = File(modelFile.parent, modelFile.name + ".config.json")
            val json = if (configFile.exists()) {
                JSONObject(configFile.readText())
            } else {
                JSONObject()
            }

            val ramGb = profile.totalRamGb
            
            // 1. Context Window (토큰)
            // 로컬 AI 프롬프트가 약 2000자 이므로 컨텍스트는 최소 1024 이상 권장
            val contextWindow = when {
                ramGb >= 16 -> 4096
                ramGb >= 12 -> 2048
                ramGb >= 8  -> 1024
                else        -> 512
            }

            // 2. Batch Size (반드시 contextWindow 보다 작거나 같아야 함)
            val batchSize = when {
                profile.recommendedBackend == HardwareAnalyzer.Backend.GPU -> 512
                ramGb >= 12 -> 256
                else -> 128
            }.coerceAtMost(contextWindow)

            // 3. GPU Layers
            val gpuLayers = if (profile.recommendedBackend == HardwareAnalyzer.Backend.GPU) {
                if (ramGb >= 12) 32 else 24
            } else 0

            json.put("n_ctx",                     contextWindow)
            json.put("n_batch",                   batchSize)
            json.put("n_ubatch",                  batchSize)
            json.put("n_gpu_layers",              gpuLayers)
            json.put("use_mmap",                  true)
            json.put("use_mlock",                 ramGb >= 12)
            json.put("n_threads",                 profile.cores.coerceAtMost(8))
            
            // 안정성을 위한 추가 설정
            json.put("cache_type_k",              "f16")
            json.put("cache_type_v",              "f16")
            json.put("grp_attn_n",                1)
            json.put("grp_attn_s",                1)

            configFile.writeText(json.toString(2))
            Log.d(TAG, "AI 최적화 정렬: ctx=$contextWindow, batch=$batchSize, gpu=$gpuLayers")
        }.onFailure {
            Log.e(TAG, "AI 설정 최적화 실패: ${it.message}")
        }
    }

    fun readConfig(modelFile: File): JSONObject? {
        return runCatching {
            val configFile = File(modelFile.parent, modelFile.name + ".config.json")
            if (configFile.exists()) JSONObject(configFile.readText()) else null
        }.getOrNull()
    }
}
