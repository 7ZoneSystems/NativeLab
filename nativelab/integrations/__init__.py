"""
External integration surface for NativeLab.

Use `IntegrationEndpoints` for Discord bots, cloud functions, CLIs, or other
automation that needs to discover NativeLab models, pipelines, labs, limits,
and the current runtime backend without importing the main window.
"""

from .endpoints import IntegrationEndpoints
from .http_endpoint import IntegrationHttpEndpoint

__all__ = ["IntegrationEndpoints", "IntegrationHttpEndpoint", "IntegrationsTab"]


def __getattr__(name):
    if name == "IntegrationsTab":
        from .tab import IntegrationsTab
        return IntegrationsTab
    raise AttributeError(name)
