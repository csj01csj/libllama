package com.aicode.studio.engine

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities

object NetworkChecker {

    enum class NetworkType { WIFI, CELLULAR, NONE }

    fun getNetworkType(context: Context): NetworkType {
        val cm  = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val nw  = cm.activeNetwork ?: return NetworkType.NONE
        val cap = cm.getNetworkCapabilities(nw) ?: return NetworkType.NONE
        return when {
            cap.hasTransport(NetworkCapabilities.TRANSPORT_WIFI)     -> NetworkType.WIFI
            cap.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET) -> NetworkType.WIFI
            cap.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) -> NetworkType.CELLULAR
            else -> NetworkType.NONE
        }
    }

    fun isConnected(context: Context) = getNetworkType(context) != NetworkType.NONE
    fun isWifi    (context: Context) = getNetworkType(context) == NetworkType.WIFI

    fun cellularWarning(context: Context, sizeGb: Float): String? {
        if (isWifi(context)) return null
        if (!isConnected(context)) return "인터넷에 연결되어 있지 않습니다."
        return "현재 모바일 데이터를 사용 중입니다.\n${sizeGb}GB 다운로드 시 요금이 발생할 수 있습니다.\n계속하시겠습니까?"
    }
}
