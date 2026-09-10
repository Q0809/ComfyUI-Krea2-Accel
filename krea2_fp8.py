"""
krea2_fp8.py
============

Convenience loader that pulls a Krea 2 diffusion checkpoint into ComfyUI with an explicit
weight dtype -- most usefully **fp8_e4m3fn**, which shrinks the 12.9B transformer from
~26 GB (bf16) to ~12 GB and lets it run on 16 GB consumer GPUs with essentially no visible
quality loss.

This is a thin wrapper around ComfyUI's native diffusion-model loader.  The exact module
that holds that loader has moved across ComfyUI releases (e.g. ``comfy_extras.nodes_model_loading``
no longer exists in recent builds such as 0.33.x), so we resolve the loader at runtime from
ComfyUI's live node registry instead of hard-coding an import path.  We prefer the
architecturally-correct ``LoadDiffusionModel`` (DiT-aware) and fall back to ``UNETLoader``,
which also loads Krea 2 checkpoints in many installs.

The model dropdown lists files from BOTH the ``diffusion_models`` and ``unet`` folders,
because different ComfyUI setups place Krea 2 safetensors in either one.
"""

import folder_paths


def _resolve_loaders():
    """Return [(name, instance), ...] of ComfyUI's live model loaders, most-preferred first.

    Reading from ``nodes.NODE_CLASS_MAPPINGS`` (populated by the time a node executes) keeps
    this version-independent: we never depend on a specific module path.
    """
    try:
        from nodes import NODE_CLASS_MAPPINGS
    except Exception as e:
        raise RuntimeError(
            "Krea2LoadDiffusionModelFP8: cannot access ComfyUI node registry "
            "(nodes.NODE_CLASS_MAPPINGS). Are you running inside ComfyUI? ({})".format(e)
        )
    loaders = []
    for key in ("LoadDiffusionModel", "UNETLoader"):
        cls = NODE_CLASS_MAPPINGS.get(key)
        if cls is not None:
            loaders.append((key, cls()))
    if not loaders:
        raise RuntimeError(
            "Krea2LoadDiffusionModelFP8: neither 'LoadDiffusionModel' nor 'UNETLoader' is "
            "registered in this ComfyUI build. Use the stock loader node directly instead."
        )
    return loaders


def _list_model_names():
    """Model filenames from both candidate folders (union, de-duplicated, sorted)."""
    names = []
    for folder_type in ("diffusion_models", "unet"):
        try:
            names.extend(folder_paths.get_filename_list(folder_type))
        except Exception:
            # Folder type not configured in this install -- skip it.
            pass
    return sorted(set(names))


class Krea2LoadDiffusionModelFP8:
    """Load a Krea 2 diffusion checkpoint with a selectable weight dtype (fp8 by default)."""

    @classmethod
    def INPUT_TYPES(cls):
        dtypes = [
            "fp8_e4m3fn",
            "fp8_e4m3fn_fast",
            "fp8_e5m2",
            "bf16",
            "fp16",
            "default",
        ]
        return {
            "required": {
                "model_name": (_list_model_names(),),
                "weight_dtype": (dtypes, {"default": "fp8_e4m3fn"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "krea2/acceleration"

    def load(self, model_name, weight_dtype):
        wd = None if weight_dtype == "default" else weight_dtype
        last_err = None
        for name, loader in _resolve_loaders():
            try:
                if hasattr(loader, "load_model"):
                    return loader.load_model(model_name=model_name, weight_dtype=wd)
                if hasattr(loader, "load_unet"):
                    return loader.load_unet(unet_name=model_name, weight_dtype=wd)
            except Exception as e:  # try the next loader (e.g. wrong folder)
                last_err = e
                continue
        raise RuntimeError(
            "Krea2LoadDiffusionModelFP8: failed to load '{}' with any available loader. "
            "({})".format(model_name, last_err)
        )
