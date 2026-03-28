package com.aicode.studio.engine

object NativeConfig {
    init {
        System.loadLibrary("nativeconfig")
    }

    @JvmStatic
    external fun disableVulkan()

    @JvmStatic
    external fun setEnv(key: String, value: String)
}
