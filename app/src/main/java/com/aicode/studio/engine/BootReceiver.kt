package com.aicode.studio.engine

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log

/**
 * 부팅 완료 수신기 — 자동 서비스 시작 비활성화.
 * (앱을 직접 열거나 Local AI 스위치를 ON 할 때만 서비스가 시작됨)
 */
class BootReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        // 부팅 시 자동 시작 안 함
    }
}
