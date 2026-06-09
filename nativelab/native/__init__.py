"""Optional native acceleration helpers.

The public API of NativeLab remains Python. Modules in this package opportunistically
route deterministic backend helper work through native code and fall back to Python
when a platform cannot build or load the native components.
"""

from .engine_helpers import (
    append_cli_sampler_args,
    build_reference_chunks,
    build_text_prompt,
    error_message,
    hf_sampler_kwargs,
    image_b64_list,
    is_context_error,
    native_core_available,
    ollama_sampler_options,
    raw_prompt_text,
    sampler_payload,
)
from .rust_model import (
    detect_family_key as detect_family_key_native,
    detect_quant_type as detect_quant_type_native,
    rust_model_available,
)

__all__ = [
    "append_cli_sampler_args",
    "build_reference_chunks",
    "build_text_prompt",
    "detect_family_key_native",
    "detect_quant_type_native",
    "error_message",
    "hf_sampler_kwargs",
    "image_b64_list",
    "is_context_error",
    "native_core_available",
    "ollama_sampler_options",
    "raw_prompt_text",
    "rust_model_available",
    "sampler_payload",
]
