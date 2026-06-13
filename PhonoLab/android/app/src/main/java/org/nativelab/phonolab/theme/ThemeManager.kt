package org.nativelab.phonolab.theme

import android.content.Context
import android.content.SharedPreferences
import androidx.appcompat.app.AppCompatDelegate

/**
 * NativeLab-inspired theme system for PhonoLab.
 * Supports light (cream) and dark (studio) palettes with PhonoLab teal accent.
 */
object ThemeManager {

    const val THEME_LIGHT = "light"
    const val THEME_DARK = "dark"

    // ── Dark Palette (NativeLab Studio Dark) ─────────────────────────
    object Dark {
        const val bg0 = "#09090d"
        const val bg1 = "#0f0f15"
        const val bg2 = "#141420"
        const val bg3 = "#1a1a28"
        const val surface = "#1e1e2e"
        const val surface2 = "#252538"
        const val accent = "#55C2A4"
        const val accent2 = "#3da88a"
        const val accentDim = "#1a3d33"
        const val txt = "#ededf5"
        const val txt2 = "#7a7a9a"
        const val txt3 = "#48485e"
        const val bdr = "#232335"
        const val bdr2 = "#2d2d45"
        const val ok = "#1cb88a"
        const val warn = "#e8971a"
        const val err = "#e84848"
        const val bubbleUser = "#0e0c26"
        const val bubbleAst = "#0c0c14"
    }

    // ── Light Palette (NativeLab Cream & Sage) ───────────────────────
    object Light {
        const val bg0 = "#fdf6f0"
        const val bg1 = "#f8ede3"
        const val bg2 = "#f2e2d4"
        const val bg3 = "#ebd6c6"
        const val surface = "#f2e2d4"
        const val surface2 = "#ebd6c6"
        const val accent = "#55C2A4"
        const val accent2 = "#3da88a"
        const val accentDim = "#d4f0e7"
        const val txt = "#1a0f0a"
        const val txt2 = "#6b4c3b"
        const val txt3 = "#b89080"
        const val bdr = "#e8cfc0"
        const val bdr2 = "#ddbfac"
        const val ok = "#15803d"
        const val warn = "#b45309"
        const val err = "#b91c1c"
        const val bubbleUser = "#fdeee4"
        const val bubbleAst = "#fdf6f0"
    }

    private lateinit var prefs: SharedPreferences

    fun init(context: Context) {
        prefs = context.getSharedPreferences("phonolab_theme", Context.MODE_PRIVATE)
    }

    fun currentTheme(): String = prefs.getString("theme", THEME_DARK) ?: THEME_DARK

    fun isDark(): Boolean = currentTheme() == THEME_DARK

    fun setTheme(theme: String) {
        prefs.edit().putString("theme", theme).apply()
        applyTheme(theme)
    }

    fun toggleTheme() {
        val new = if (isDark()) THEME_LIGHT else THEME_DARK
        setTheme(new)
    }

    fun applyTheme(theme: String = currentTheme()) {
        when (theme) {
            THEME_DARK -> AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_YES)
            THEME_LIGHT -> AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_NO)
        }
    }

    /** Get the active palette as a map for programmatic access. */
    fun palette(): Map<String, String> {
        return if (isDark()) darkPalette() else lightPalette()
    }

    private fun darkPalette() = mapOf(
        "bg0" to Dark.bg0, "bg1" to Dark.bg1, "bg2" to Dark.bg2, "bg3" to Dark.bg3,
        "surface" to Dark.surface, "surface2" to Dark.surface2,
        "accent" to Dark.accent, "accent2" to Dark.accent2, "accentDim" to Dark.accentDim,
        "txt" to Dark.txt, "txt2" to Dark.txt2, "txt3" to Dark.txt3,
        "bdr" to Dark.bdr, "bdr2" to Dark.bdr2,
        "ok" to Dark.ok, "warn" to Dark.warn, "err" to Dark.err,
        "bubbleUser" to Dark.bubbleUser, "bubbleAst" to Dark.bubbleAst,
    )

    private fun lightPalette() = mapOf(
        "bg0" to Light.bg0, "bg1" to Light.bg1, "bg2" to Light.bg2, "bg3" to Light.bg3,
        "surface" to Light.surface, "surface2" to Light.surface2,
        "accent" to Light.accent, "accent2" to Light.accent2, "accentDim" to Light.accentDim,
        "txt" to Light.txt, "txt2" to Light.txt2, "txt3" to Light.txt3,
        "bdr" to Light.bdr, "bdr2" to Light.bdr2,
        "ok" to Light.ok, "warn" to Light.warn, "err" to Light.err,
        "bubbleUser" to Light.bubbleUser, "bubbleAst" to Light.bubbleAst,
    )
}
