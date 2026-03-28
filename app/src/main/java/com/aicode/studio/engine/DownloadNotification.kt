package com.aicode.studio.engine

import android.content.Context
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat

object DownloadNotification {

    private const val CHANNEL_ID = "model_download"
    private const val NOTIF_ID   = 2001

    fun show(context: Context, modelName: String, percent: Int) {
        val notif = NotificationCompat.Builder(context, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle("모델 다운로드")
            .setContentText("$modelName — ${percent}%")
            .setProgress(100, percent, percent == 0)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setOnlyAlertOnce(true)
            .build()
        runCatching { NotificationManagerCompat.from(context).notify(NOTIF_ID, notif) }
    }

    fun update(context: Context, modelName: String, percent: Int, speedKbps: Float) {
        val speedText = when {
            speedKbps > 1024 -> "${"%.1f".format(speedKbps / 1024)} MB/s"
            else             -> "${speedKbps.toInt()} KB/s"
        }
        val notif = NotificationCompat.Builder(context, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle("$modelName 다운로드 중")
            .setContentText("${percent}%  ·  $speedText")
            .setSubText("${percent}% 완료")
            .setProgress(100, percent, false)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setOnlyAlertOnce(true)
            .build()
        runCatching { NotificationManagerCompat.from(context).notify(NOTIF_ID, notif) }
    }

    fun showComplete(context: Context, modelName: String) {
        val notif = NotificationCompat.Builder(context, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_download_done)
            .setContentTitle("다운로드 완료")
            .setContentText("$modelName 사용 준비됨")
            .setAutoCancel(true)
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .build()
        runCatching { NotificationManagerCompat.from(context).notify(NOTIF_ID + 1, notif) }
    }

    fun cancel(context: Context) {
        runCatching { NotificationManagerCompat.from(context).cancel(NOTIF_ID) }
    }

    fun needsPermission(): Boolean = Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU
}
