package org.nativelab.phonolab.api

import android.content.Context
import android.content.SharedPreferences
import java.net.NetworkInterface
import java.util.UUID

/**
 * API server configuration with key generation and persistence.
 * Mirrors NativeLab's ApiServerConfig for network compatibility.
 */
data class ApiConfig(
    val host: String = "0.0.0.0",
    val port: Int = 8787,
    val protocol: String = "both",       // "openai", "anthropic", "both"
    val requireApiKey: Boolean = true,
    val localApiKey: String = "",
    val lanApiKey: String = "",
) {
    val localBaseUrl: String get() = "http://127.0.0.1:$port/v1"
    val lanBaseUrl: String get() = "http://${detectLanIp()}:$port/v1"

    fun supportsOpenAi(): Boolean = protocol in listOf("openai", "both")
    fun supportsAnthropic(): Boolean = protocol in listOf("anthropic", "both")

    fun save(prefs: SharedPreferences) {
        prefs.edit()
            .putString("host", host)
            .putInt("port", port)
            .putString("protocol", protocol)
            .putBoolean("require_api_key", requireApiKey)
            .putString("local_api_key", localApiKey)
            .putString("lan_api_key", lanApiKey)
            .apply()
    }

    companion object {
        private const val PREFS_NAME = "phonolab_api_server"

        fun generateApiKey(): String = "nl-${UUID.randomUUID().toString().replace("-", "").take(32)}"

        fun detectLanIp(): String {
            try {
                for (iface in NetworkInterface.getNetworkInterfaces()) {
                    if (!iface.isUp || iface.isLoopback) continue
                    for (addr in iface.inetAddresses) {
                        val ip = addr.hostAddress ?: continue
                        if (ip.contains(".") && !ip.startsWith("127.")) {
                            return ip
                        }
                    }
                }
            } catch (_: Exception) { }
            return "127.0.0.1"
        }

        fun load(context: Context): ApiConfig {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            val localKey = prefs.getString("local_api_key", "") ?: ""
            val lanKey = prefs.getString("lan_api_key", "") ?: ""
            val config = ApiConfig(
                host = prefs.getString("host", "0.0.0.0") ?: "0.0.0.0",
                port = prefs.getInt("port", 8787),
                protocol = prefs.getString("protocol", "both") ?: "both",
                requireApiKey = prefs.getBoolean("require_api_key", true),
                localApiKey = if (localKey.isNotEmpty()) localKey else generateApiKey(),
                lanApiKey = if (lanKey.isNotEmpty()) lanKey else generateApiKey(),
            )
            // Save if keys were generated
            if (localKey.isEmpty() || lanKey.isEmpty()) {
                config.save(prefs)
            }
            return config
        }
    }
}
