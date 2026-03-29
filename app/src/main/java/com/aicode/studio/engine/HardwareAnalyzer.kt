package com.aicode.studio.engine

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.util.Log
import java.io.File
import java.util.regex.Pattern

/**
 * 기기 하드웨어를 분석해 최적 추론 백엔드와
 * 사용 가능한 모델 크기를 결정한다.
 *
 * 우선순위: GPU(Vulkan/OpenCL) > NPU > DSP > CPU
 */
object HardwareAnalyzer {

    private const val TAG = "HardwareAnalyzer"

    data class HardwareProfile(
        val socName           : String,
        val totalRamGb        : Int,
        val availRamGb        : Int,
        val cores             : Int,
        // CPU Features (ARM Specific)
        val hasFp16           : Boolean,
        val hasDotProd        : Boolean,
        val hasSve            : Boolean,
        val hasI8mm           : Boolean,
        // Accelerators
        val hasNpu            : Boolean,
        val npuName           : String,
        val hasDsp            : Boolean,
        val dspVersion        : String,
        val hasVulkan         : Boolean,
        val vulkanVersion     : String,
        val gpuRenderer       : String,
        val hasNnapi          : Boolean,
        val recommendedBackend: Backend,
        val maxAllowedRamGb   : Int
    )

    enum class Backend(val displayName: String) {
        GPU("Vulkan/GPU"),
        NPU("NNAPI (NPU)"),
        DSP("Hexagon DSP"),
        CPU("CPU (ARM NEON/DotProd)")
    }

    fun analyze(context: Context): HardwareProfile {
        val soc    = detectSoC()
        val ram    = getRamInfo(context)
        val cpu    = detectCpuFeatures()
        val npu    = detectNpu(soc)
        val dsp    = detectDsp(soc)
        val vulkan = detectVulkan()
        val gpuRenderer = detectGpuRenderer()
        val nnapi  = Build.VERSION.SDK_INT >= Build.VERSION_CODES.P

        // Backend Recommendation Logic
        val backend = when {
            vulkan.first && !isBuggyVulkan(soc, gpuRenderer) -> Backend.GPU
            napiEnabled() && nnapi -> Backend.NPU
            dsp.first -> Backend.DSP
            else -> Backend.CPU
        }

        val ratio = when {
            ram.first >= 10 -> 0.55   // 12GB 폰 (실제 ~11GB 리포트) → maxRam ~6GB
            ram.first >= 7  -> 0.50   // 8GB 폰 (실제 ~7GB 리포트)   → maxRam ~3GB
            else            -> 0.40
        }
        val maxRam = (ram.first * ratio).toInt().coerceAtLeast(1)

        return HardwareProfile(
            socName            = soc,
            totalRamGb         = ram.first,
            availRamGb         = ram.second,
            cores              = Runtime.getRuntime().availableProcessors(),
            hasFp16            = cpu.contains("fp16") || cpu.contains("fphp"),
            hasDotProd         = cpu.contains("dotprod") || cpu.contains("asimddp"),
            hasSve             = cpu.contains("sve"),
            hasI8mm            = cpu.contains("i8mm"),
            hasNpu             = npu.first,
            npuName            = npu.second,
            hasDsp             = dsp.first,
            dspVersion         = dsp.second,
            hasVulkan          = vulkan.first,
            vulkanVersion      = vulkan.second,
            gpuRenderer        = gpuRenderer,
            hasNnapi           = nnapi,
            recommendedBackend = backend,
            maxAllowedRamGb    = maxRam
        )
    }

    private fun detectSoC(): String {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val soc = Build.SOC_MODEL
            if (soc.isNotBlank() && soc != Build.UNKNOWN) return soc
        }
        return try {
            File("/proc/cpuinfo").readLines()
                .firstOrNull { it.startsWith("Hardware") || it.startsWith("Model name") }
                ?.substringAfter(":")?.trim()
                ?: "${Build.MANUFACTURER} ${Build.HARDWARE}"
        } catch (_: Exception) {
            "${Build.MANUFACTURER} ${Build.HARDWARE}"
        }
    }

    private fun getRamInfo(context: Context): Pair<Int, Int> {
        val am   = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val info = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        val total = (info.totalMem / (1024.0 * 1024 * 1024)).toInt()
        val avail = (info.availMem / (1024.0 * 1024 * 1024)).toInt()
        return Pair(total, avail)
    }

    private fun detectCpuFeatures(): Set<String> {
        val features = mutableSetOf<String>()
        try {
            File("/proc/cpuinfo").forEachLine { line ->
                if (line.startsWith("Features") || line.startsWith("flags")) {
                    val parts = line.split(":")
                    if (parts.size >= 2) {
                        features.addAll(parts[1].trim().split(" ").filter { it.isNotEmpty() })
                    }
                }
            }
        } catch (_: Exception) {}
        return features
    }

    private fun detectNpu(soc: String): Pair<Boolean, String> {
        val socLower = soc.lowercase()
        if (socLower.contains("snapdragon") || socLower.contains("sm8") ||
            socLower.contains("qcom") || Build.MANUFACTURER.equals("qualcomm", true)) {
            if (File("/dev/nnapi-qnn-hta").exists() ||
                File("/sys/class/qcom_subsystem/adsp/name").exists()) {
                return Pair(true, "Qualcomm Hexagon NPU")
            }
        }
        if (socLower.contains("exynos") || Build.MANUFACTURER.equals("samsung", true)) {
            if (File("/dev/npumgr").exists() ||
                File("/sys/class/dsp/vertex0/name").exists()) {
                return Pair(true, "Samsung Exynos NPU")
            }
        }
        if (socLower.contains("mediatek") || socLower.contains("mt") ||
            Build.MANUFACTURER.equals("mediatek", true)) {
            if (File("/dev/mdla0").exists() || File("/dev/apusys").exists()) {
                return Pair(true, "MediaTek APU")
            }
        }
        if (socLower.contains("tensor") || Build.MANUFACTURER.equals("google", true)) {
            return Pair(true, "Google Tensor TPU")
        }
        return Pair(false, "없음")
    }

    private fun detectDsp(soc: String): Pair<Boolean, String> {
        val socLower = soc.lowercase()
        if (!socLower.contains("snapdragon") && !socLower.contains("sm8") &&
            !socLower.contains("qcom")) {
            return Pair(false, "해당없음")
        }
        val htpVersion = try {
            File("/sys/class/qcom_subsystem/cdsp/name").readText().trim()
        } catch (_: Exception) { "" }

        if (File("/dev/cdsp").exists() || File("/dev/fastrpc-cdsp").exists() ||
            htpVersion.isNotEmpty()) {
            val version = when {
                soc.contains("8 Gen 3") || soc.contains("8650") -> "HTP v75"
                soc.contains("8 Gen 2") || soc.contains("8550") -> "HTP v73"
                soc.contains("8 Gen 1") || soc.contains("8450") -> "HTP v69"
                soc.contains("888")     -> "HTP v68"
                else                    -> "Hexagon DSP"
            }
            return Pair(true, version)
        }
        return Pair(false, "해당없음")
    }

    private fun detectVulkan(): Pair<Boolean, String> {
        if (Build.VERSION.SDK_INT < 24) return Pair(false, "미지원")
        return try {
            System.loadLibrary("vulkan")
            val ver = when {
                Build.VERSION.SDK_INT >= 31 -> "1.3"
                Build.VERSION.SDK_INT >= 28 -> "1.1"
                else                        -> "1.0"
            }
            Pair(true, ver)
        } catch (_: UnsatisfiedLinkError) {
            Pair(false, "드라이버 없음")
        } catch (_: Exception) {
            Pair(false, "로드 실패")
        }
    }

    private fun detectGpuRenderer(): String {
        // Simplified detection via Build.HARDWARE if EGL is too much, 
        // but let's try to look for Adreno/Mali in system properties
        return try {
            val renderer = System.getProperty("ro.hardware.egl") ?: ""
            if (renderer.isNotEmpty()) return renderer
            
            // Fallback to searching in cpuinfo or build props
            if (Build.HARDWARE.lowercase().contains("qcom")) "Adreno"
            else if (Build.HARDWARE.lowercase().contains("mali")) "Mali"
            else "Generic"
        } catch (_: Exception) { "Unknown" }
    }

    private fun isBuggyVulkan(soc: String, renderer: String): Boolean {
        // Vulkan crash protection is now handled at the JNI layer (safe_backend_init with
        // sigsetjmp/siglongjmp).  No devices are pre-blocked here; the native layer will
        // auto-detect and fall back to CPU if the driver crashes.
        return false
    }

    private fun napiEnabled(): Boolean = false // Default disabled unless stable

    fun allowedModels(profile: HardwareProfile): Set<String> {
        return InferenceConfig.ALL_MODELS
            .filter { it.minRamGb <= profile.maxAllowedRamGb }
            .map { it.id }
            .toSet()
    }

    fun HardwareProfile.toReport(): String = buildString {
        appendLine("┌─ 시스템 상세 분석 ─────────────────")
        appendLine("│ SoC      : $socName ($cores Cores)")
        appendLine("│ RAM      : ${totalRamGb}GB (가용 ${availRamGb}GB)")
        appendLine("├─ CPU 가속 ──────────────────────────")
        appendLine("│ FP16     : ${if (hasFp16) "✅" else "❌"}")
        appendLine("│ DotProd  : ${if (hasDotProd) "✅" else "❌"}")
        appendLine("│ I8MM     : ${if (hasI8mm) "✅" else "❌"}")
        appendLine("│ SVE      : ${if (hasSve) "✅" else "❌"}")
        appendLine("├─ 가속기 ────────────────────────────")
        appendLine("│ GPU      : ${if (hasVulkan) "✅ $gpuRenderer (Vulkan $vulkanVersion)" else "❌"}")
        appendLine("│ NPU      : ${if (hasNpu) "✅ $npuName" else "❌"}")
        appendLine("│ DSP      : ${if (hasDsp) "✅ $dspVersion" else "❌"}")
        appendLine("├────────────────────────────────────")
        appendLine("│ 추천 백엔드: ${recommendedBackend.displayName}")
        appendLine("│ AI 할당 RAM: ${maxAllowedRamGb}GB")
        append("└────────────────────────────────────")
    }
}
