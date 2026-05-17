"""Shared long timeout constants.

Keep this module lightweight so CLI tools, setup scripts, and integration
runners can use the same timeout policy without importing the GUI stack.
"""

LONG_TIMEOUT_SECONDS = 24 * 60 * 60
LONG_TIMEOUT_MS = LONG_TIMEOUT_SECONDS * 1000
LONG_TIMEOUT_NONE = None

