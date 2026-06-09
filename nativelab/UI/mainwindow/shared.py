"""Shared imports for the split MainWindow modules.

This mirrors the historical nativelab.main import surface so extracted mixins can
stay behavior-compatible while the large window class is broken into feature files.
"""
from nativelab.imports.import_global import *
from nativelab.GlobalConfig.config_global import *
from nativelab.Model.model_global import *
from nativelab.UI.UI_global import *
from nativelab.Prefrences.prefrence_global import *
from nativelab.Server.server_global import *
from nativelab.core.streamer_global import *
from nativelab.components.components_global import *
from nativelab.core.engine_global import *
from nativelab.core.auto_setup import (
    AutoSetupWorker,
    auto_setup_resumable,
    decline_auto_setup,
    normalize_setup_backend,
    read_auto_setup_state,
    user_needs_auto_setup,
)
from nativelab.core.context_meter import context_meter
from nativelab.core.model_loaders import ApiLoaderThread, ModelLoaderThread
from nativelab.codeparser.codeparser_global import *
from nativelab.pipelinebuilder.pipe_global import *
from nativelab.UI.icons import (
    add_menu_action,
    icon,
    icon_size,
    refresh_widget_icons,
    role_icon,
    set_button_icon,
    set_label_icon,
    status_icon,
    set_status_label,
)
from nativelab.UI.buildUI import palette_rgba
from nativelab.UI.qt_workers import stop_worker, stop_worker_attrs
from nativelab.labs import LabEndpoints, LabsTab
from nativelab.integrations import IntegrationEndpoints, IntegrationsTab
from nativelab.skill import active_skill_context, ensure_builtin_edit_skill
from nativelab.skill.tab import SkillsTab
from nativelab.Server.ollama_helpers import normalize_ollama_exception, normalize_ollama_host
from nativelab.api_server import ApiServerTab
