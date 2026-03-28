#include <jni.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <android/log.h>

#define TAG "NativeConfig"

// llama.cpp는 일부 설정을 환경 변수로 읽을 수 있습니다.
JNIEXPORT void JNICALL
Java_com_aicode_studio_engine_NativeConfig_disableVulkan(JNIEnv *env, jclass clazz) {
    setenv("GGML_VK_DISABLE", "1", 1);
}

JNIEXPORT void JNICALL
Java_com_aicode_studio_engine_NativeConfig_setEnv(JNIEnv *env, jclass clazz, jstring key, jstring value) {
    const char *nativeKey = (*env)->GetStringUTFChars(env, key, 0);
    const char *nativeValue = (*env)->GetStringUTFChars(env, value, 0);

    // 1은 덮어쓰기 허용
    setenv(nativeKey, nativeValue, 1);

    // llama.cpp 호환을 위해 추가 접두사도 시도 (엔진에 따라 다름)
    if (strstr(nativeKey, "GGML_") == nativeKey) {
         // 이미 GGML_ 접두사가 있으면 그대로
    } else {
         // LLAMA_ 접두사도 추가 셋팅
         char llamaKey[256];
         snprintf(llamaKey, sizeof(llamaKey), "LLAMA_%s", nativeKey);
         setenv(llamaKey, nativeValue, 1);
    }

    __android_log_print(ANDROID_LOG_DEBUG, TAG, "setenv %s=%s", nativeKey, nativeValue);

    (*env)->ReleaseStringUTFChars(env, key, nativeKey);
    (*env)->ReleaseStringUTFChars(env, value, nativeValue);
}
