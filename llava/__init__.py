"""LLaVA model package.

The released 3DRS code imported the Llama implementation unconditionally.
That makes importing the package fail when an optional Llama dependency is
missing, even when the requested checkpoint is the Qwen2 LLaVA-Video model.
Keep the legacy symbol when it is available, but allow model-specific lazy
imports (for example ``llava_qwen``) to proceed independently.
"""

try:
    from .model import LlavaLlamaForCausalLM
except (ImportError, ModuleNotFoundError):
    LlavaLlamaForCausalLM = None
