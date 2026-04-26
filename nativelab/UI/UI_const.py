# ── Theme ─────────────────────────────────────────────────────────────────────
CURRENT_THEME = "light"   # "dark" | "light"  ← toggle here
# ── colour palettes ───────────────────────────────────────────────────────────
C_DARK = {
    "bg0":      "#09090d",
    "bg1":      "#0f0f15",
    "bg2":      "#141420",
    "bg3":      "#1a1a28",
    "surface":  "#1e1e2e",
    "surface2": "#252538",
    "highlight":"rgba(105,92,235,0.13)",
    "acc":      "#695ceb",
    "acc2":     "#9d93f5",
    "acc_dim":  "rgba(105,92,235,0.22)",
    "usr":      "#0e0c26",
    "ast":      "#0c0c14",
    "rsn":      "#090d1c",
    "cod":      "#090c0a",
    "txt":      "#ededf5",
    "txt2":     "#7a7a9a",
    "txt3":     "#48485e",
    "bdr":      "#232335",
    "bdr2":     "#2d2d45",
    "ok":       "#1cb88a",
    "warn":     "#e8971a",
    "err":      "#e84848",
    "glow":     "#4b3be0",
    "pipeline": "#18b0ca",
}

C_LIGHT = {
    "bg0":      "#fdf6f0",
    "bg1":      "#f8ede3",
    "bg2":      "#f2e2d4",
    "bg3":      "#ebd6c6",
    "surface":  "#f2e2d4",
    "surface2": "#ebd6c6",
    "highlight":"rgba(194,65,12,0.07)",
    "acc":      "#c2410c",
    "acc2":     "#9a3412",
    "acc_dim":  "#fde8d8",
    "usr":      "#fdeee4",
    "ast":      "#fdf6f0",
    "rsn":      "#f0faf8",
    "cod":      "#f3faf0",
    "txt":      "#1a0f0a",
    "txt2":     "#6b4c3b",
    "txt3":     "#b89080",
    "bdr":      "#e8cfc0",
    "bdr2":     "#ddbfac",
    "ok":       "#15803d",
    "warn":     "#b45309",
    "err":      "#b91c1c",
    "glow":     "#9a3412",
    "pipeline": "#0e7490",
}
# Active palette — driven by CURRENT_THEME set at top of file
C = {}

C = {}

def set_theme(theme: str, light_custom=None, dark_custom=None):
    global CURRENT_THEME
    CURRENT_THEME = theme

    if light_custom:
        C_LIGHT.update(light_custom)
    if dark_custom:
        C_DARK.update(dark_custom)

    C.clear()
    C.update(C_LIGHT if theme == "light" else C_DARK)

    # FORCE UI REFRESH (THIS FIXES CANVAS)
    try:
        from PyQt6.QtWidgets import QApplication
        for w in QApplication.allWidgets():
            if hasattr(w, "refresh_theme"):
                w.refresh_theme()
            elif hasattr(w, "update"):
                w.update()        
    except Exception:
        pass

set_theme(CURRENT_THEME)


# Typography constants
FONT_UI   = "'Inter','Segoe UI','SF Pro Display',system-ui,-apple-system,sans-serif"
FONT_MONO = "'JetBrains Mono','Fira Code','Cascadia Code','Consolas',monospace"