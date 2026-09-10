"""
nodes.py
========

ComfyUI 自定义节点，用于加速本地 Krea 2 扩散模型：

  * Krea2CachePatch            — TeaCache 风格残差缓存（显示名 "Krea2 Cache Patch (TeaCache)"）
  * Krea2CacheStats            — 打印/查看缓存命中统计（显示名 "Krea2 Cache Stats"）
  * Krea2LoadDiffusionModelFP8 — 以 fp8 等精度加载 Krea 2 权重（可选）

实现见 krea2_cache.py / krea2_fp8.py，用法见 README.md。
"""

import numpy as np

try:  # 兼容两种导入方式：作为包（相对导入）或目录在 sys.path 上（绝对导入）
    from .krea2_cache import apply_krea2_cache, get_krea2_cache_stats
except ImportError:  # pragma: no cover
    from krea2_cache import apply_krea2_cache, get_krea2_cache_stats

# FP8 加载器是可选功能：即使它因为 ComfyUI 版本差异导入失败，
# 也不能影响缓存节点（ComfyUI 会静默吞掉整个包的导入异常）。
try:
    try:
        from .krea2_fp8 import Krea2LoadDiffusionModelFP8
    except ImportError:
        from krea2_fp8 import Krea2LoadDiffusionModelFP8
except Exception as _e:  # pragma: no cover
    print(f"[ComfyUI-Krea2-Accel] FP8 加载器不可用，已跳过该节点: {_e}")
    Krea2LoadDiffusionModelFP8 = None


class Krea2CachePatch:
    """给 Krea 2 (SingleStreamDiT) 的 MODEL 注入 TeaCache 残差缓存。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enable": ("BOOLEAN", {"default": True}),
                "rel_l1_thresh": (
                    "FLOAT",
                    {
                        "default": 0.30,
                        "min": 0.0,
                        "max": 3.0,
                        "step": 0.01,
                        "tooltip": "相对 L1 变化阈值。越大跳得越多 → 越快但画质风险越高。"
                                   "先用默认的 0.30 跑一张，看 ComfyUI 控制台打印的"
                                   "'逐步相对L1'，再按它给的建议值微调。",
                    },
                ),
                "start_percent": (
                    "FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "end_percent": (
                    "FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "max_skip_steps": (
                    "INT",
                    {
                        "default": 3,
                        "min": 0,
                        "max": 10,
                        "step": 1,
                        "tooltip": "最多连续复用多少步。防止长时间不计算导致画质崩坏。步数少时可设 3。",
                    },
                ),
                "cache_device": (["default", "cpu", "gpu"], {"default": "default"}),
                "coefficients": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "可选：TeaCache 四次多项式系数，5 个逗号分隔数字，"
                                   "例如 '-450,280,-45,3.2,-0.02'。留空=原始相对 L1 模式（推荐）。",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "krea2/acceleration"

    def apply(self, model, enable, rel_l1_thresh, start_percent, end_percent,
              max_skip_steps, cache_device, coefficients):
        coef = None
        coefficients = (coefficients or "").strip()
        if coefficients:
            try:
                nums = [float(x) for x in coefficients.split(",") if x.strip() != ""]
                if len(nums) != 5:
                    raise ValueError("需要恰好 5 个系数")
                coef = np.poly1d(nums)
            except Exception as e:
                raise RuntimeError(
                    f"Krea2CachePatch: coefficients 无效 '{coefficients}': {e}"
                )

        patched = apply_krea2_cache(
            model,
            enable=enable,
            thresh=rel_l1_thresh,
            start_p=start_percent,
            end_p=end_percent,
            max_skip=max_skip_steps,
            cache_device=cache_device,
            coefficients=coef,
        )
        return (patched,)


class Krea2CacheStats:
    """查看 Krea2 Cache Patch 的命中率（确认加速是否真的生效）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "stats"
    CATEGORY = "krea2/acceleration"
    OUTPUT_NODE = True

    def stats(self, model):
        return (get_krea2_cache_stats(model),)


# ---------------------------------------------------------------------------
# 注册
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "Krea2CachePatch": Krea2CachePatch,
    "Krea2CacheStats": Krea2CacheStats,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2CachePatch": "Krea2 Cache Patch (TeaCache)",
    "Krea2CacheStats": "Krea2 Cache Stats",
}

if Krea2LoadDiffusionModelFP8 is not None:
    NODE_CLASS_MAPPINGS["Krea2LoadDiffusionModelFP8"] = Krea2LoadDiffusionModelFP8
    NODE_DISPLAY_NAME_MAPPINGS["Krea2LoadDiffusionModelFP8"] = "Krea2 Load Model (FP8)"
