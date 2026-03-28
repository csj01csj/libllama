package com.aicode.studio.util

import android.content.Context

object PrefsManager {
    private const val P = "aide_prefs"
    private fun p(c: Context) = c.getSharedPreferences(P, Context.MODE_PRIVATE)

    fun saveApiKey(c: Context, k: String) = p(c).edit().putString("api_key", k).apply()
    fun getApiKey(c: Context): String = p(c).getString("api_key", "") ?: ""

    fun saveModel(c: Context, m: String) = p(c).edit().putString("model", m).apply()
    fun getModel(c: Context): String = p(c).getString("model", "gemini-2.0-flash") ?: "gemini-2.0-flash"

    fun saveLastProject(c: Context, path: String) = p(c).edit().putString("last_project", path).apply()
    fun getLastProject(c: Context): String = p(c).getString("last_project", "") ?: ""

    fun saveFontSize(c: Context, s: Float) = p(c).edit().putFloat("font_size", s).apply()
    fun getFontSize(c: Context): Float = p(c).getFloat("font_size", 13f)

    // 프로젝트별 탭 상태 (JSON Array string)
    fun saveProjectTabs(c: Context, projectName: String, tabs: String) = 
        p(c).edit().putString("tabs_$projectName", tabs).apply()
    fun getProjectTabs(c: Context, projectName: String): String = 
        p(c).getString("tabs_$projectName", "[]") ?: "[]"

    // 다중 API 설정 (JSON Array of {key, model})
    fun saveApiConfigs(c: Context, json: String) = p(c).edit().putString("api_configs", json).apply()
    fun getApiConfigs(c: Context): String = p(c).getString("api_configs", "[]") ?: "[]"

    // 키스토어별 비밀번호 기억
    fun saveKeystorePass(c: Context, path: String, pass: String) =
        p(c).edit().putString("kspass_$path", pass).apply()
    fun getKeystorePass(c: Context, path: String): String =
        p(c).getString("kspass_$path", "") ?: ""

    // Local AI 모드
    fun saveLocalAIMode(c: Context, enabled: Boolean) = p(c).edit().putBoolean("local_ai_mode", enabled).apply()
    fun getLocalAIMode(c: Context): Boolean = p(c).getBoolean("local_ai_mode", false)
}