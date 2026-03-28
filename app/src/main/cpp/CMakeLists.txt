cmake_minimum_required(VERSION 3.22.1)
project("aicode_native")

# ── Prebuilt libllama.so and its dependencies ─────────────────────────────────
# These are copied from the llamatik 0.14.0 AAR into jniLibs/arm64-v8a/

set(PREBUILT_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}")

add_library(llama         SHARED IMPORTED)
add_library(ggml          SHARED IMPORTED)
add_library(ggml_base     SHARED IMPORTED)
add_library(ggml_cpu      SHARED IMPORTED)
add_library(omp           SHARED IMPORTED)

set_target_properties(llama     PROPERTIES IMPORTED_LOCATION "${PREBUILT_LIB_DIR}/libllama.so")
set_target_properties(ggml      PROPERTIES IMPORTED_LOCATION "${PREBUILT_LIB_DIR}/libggml.so")
set_target_properties(ggml_base PROPERTIES IMPORTED_LOCATION "${PREBUILT_LIB_DIR}/libggml-base.so")
set_target_properties(ggml_cpu  PROPERTIES IMPORTED_LOCATION "${PREBUILT_LIB_DIR}/libggml-cpu.so")
set_target_properties(omp       PROPERTIES IMPORTED_LOCATION "${PREBUILT_LIB_DIR}/libomp.so")

# ── Our direct JNI bridge ──────────────────────────────────────────────────────
add_library(llama_direct SHARED llama_direct_jni.cpp)

target_include_directories(llama_direct PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

find_library(log-lib log)
find_library(android-lib android)

target_link_libraries(llama_direct
    llama
    ggml
    ggml_base
    ggml_cpu
    omp
    ${log-lib}
    ${android-lib})

# ── Existing NativeConfig shim ────────────────────────────────────────────────
add_library(nativeconfig SHARED native_config.c)
target_link_libraries(nativeconfig ${log-lib})
