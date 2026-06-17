/**
 * JNI wrapper for executing binaries on Android.
 *
 * Forks a child process and calls execve() from native code.
 * The child inherits the envp array, which MUST contain LD_LIBRARY_PATH
 * pointing to nativeLibraryDir where all .so dependencies live.
 */
#include <jni.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <errno.h>
#include <string.h>
#include <string>
#include <vector>
#include <android/log.h>

#define TAG "LlamaRunner"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

extern "C" JNIEXPORT jint JNICALL
Java_org_nativelab_phonolab_LlamaCppManager_nativeExec(
        JNIEnv *env, jobject,
        jstring jBinary,
        jobjectArray jArgs,
        jobjectArray jEnv) {

    const char *binary = env->GetStringUTFChars(jBinary, nullptr);

    // argv[0] MUST be the binary itself — prepend it explicitly
    std::vector<std::string> args_storage;
    args_storage.emplace_back(binary);  // argv[0] = binary path (required by execve)

    int argc = env->GetArrayLength(jArgs);
    for (int i = 0; i < argc; i++) {
        auto js = (jstring) env->GetObjectArrayElement(jArgs, i);
        const char *s = env->GetStringUTFChars(js, nullptr);
        args_storage.emplace_back(s);
        env->ReleaseStringUTFChars(js, s);
        env->DeleteLocalRef(js);
    }

    std::vector<char *> argv;
    for (auto &s : args_storage) argv.push_back(const_cast<char *>(s.c_str()));
    argv.push_back(nullptr);

    // Build envp from Kotlin-supplied env array
    std::vector<std::string> env_storage;
    int envc = env->GetArrayLength(jEnv);
    for (int i = 0; i < envc; i++) {
        auto js = (jstring) env->GetObjectArrayElement(jEnv, i);
        const char *s = env->GetStringUTFChars(js, nullptr);
        env_storage.emplace_back(s);
        env->ReleaseStringUTFChars(js, s);
        env->DeleteLocalRef(js);
    }
    std::vector<char *> envp;
    for (auto &s : env_storage) envp.push_back(const_cast<char *>(s.c_str()));
    envp.push_back(nullptr);

    // Log everything before fork so we can diagnose in logcat
    LOGD("execve binary: %s", binary);
    for (size_t i = 0; i < args_storage.size(); i++)
        LOGD("  argv[%zu]: %s", i, args_storage[i].c_str());
    for (auto &s : env_storage)
        LOGD("  env: %s", s.c_str());

    pid_t pid = fork();
    if (pid == 0) {
        // Child process — exec the binary with full env
        execve(binary, argv.data(), envp.data());
        // If we get here, execve failed
        LOGE("execve FAILED for %s: errno=%d (%s)", binary, errno, strerror(errno));
        _exit(127);
    }

    env->ReleaseStringUTFChars(jBinary, binary);

    if (pid < 0) {
        LOGE("fork() failed: errno=%d (%s)", errno, strerror(errno));
        return -1;
    }

    LOGI("Spawned PID: %d", pid);
    return pid;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_org_nativelab_phonolab_LlamaCppManager_nativeKill(
        JNIEnv *, jobject, jint pid) {
    int result = kill((pid_t)pid, SIGTERM);
    LOGD("kill SIGTERM pid=%d result=%d", pid, result);
    return result == 0 ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_org_nativelab_phonolab_LlamaCppManager_nativeKillForcibly(
        JNIEnv *, jobject, jint pid) {
    int result = kill((pid_t)pid, SIGKILL);
    LOGD("kill SIGKILL pid=%d result=%d", pid, result);
    return result == 0 ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jint JNICALL
Java_org_nativelab_phonolab_LlamaCppManager_nativeWaitPid(
        JNIEnv *, jobject, jint pid) {
    int status = 0;
    pid_t result = waitpid((pid_t)pid, &status, WNOHANG);
    if (result > 0) {
        if (WIFEXITED(status))
            LOGD("pid=%d exited with code=%d", pid, WEXITSTATUS(status));
        else if (WIFSIGNALED(status))
            LOGD("pid=%d killed by signal=%d", pid, WTERMSIG(status));
    }
    return (jint)result;
}
