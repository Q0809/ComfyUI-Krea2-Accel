"""
krea2_cache.py
==============

TeaCache-style 残差缓存加速，面向 ComfyUI 里的 Krea 2 扩散模型
(``comfy.ldm.krea2.model.SingleStreamDiT``)。

Krea 2 的结构（已按 ComfyUI 官方源码核对）
------------------------------------------
``SingleStreamDiT._forward`` 把文本 token 和图像 token 拼成一条序列 ``combined``，
然后顺序通过 ``self.blocks``（28 个 ``SingleStreamBlock``，宽度 6144）::

    combined = torch.cat((context, img), dim=1)
    for i, block in enumerate(self.blocks):
        combined = block(combined, tvec, freqs, None,
                         timestep_zero_index=..., transformer_options=...)

每个 ``SingleStreamBlock.forward(x, vec, freqs, mask, timestep_zero_index, transformer_options)``
都是残差形式：返回 ``x + Δ``（AdaLN 调制的 attention + SwiGLU MLP）。

加速原理
--------
相邻采样步之间，整个 block stack 的"总更新量" ``R = out_N - in_0`` 变化很小。
所以：

* **命中缓存**：block 0 直接返回 ``x + R_cached``，block 1..N-1 全部退化成恒等映射，
  最终 stack 输出 = ``x + R_cached``。28 个 block 的 attention/MLP 全部省掉。
* **未命中**：正常跑完 28 个 block，在最后一个 block 处刷新 ``R``。

是否命中由 block 0 的 **AdaLN 调制输入** ``m = (1 + prescale) * prenorm(x) + preshift``
与上一步的相对 L1 距离决定（累加后与阈值比较），这就是 TeaCache 的判据。

为什么这样打补丁是安全的
------------------------
* 不重写 ``_forward``，不碰任何 tensor 的 shape —— 所有 shape 仍然由 ComfyUI 原生代码处理。
* ``_forward`` 只包一层"记录当前 timestep"的壳，用于算采样进度（start/end percent）。
* 每个 block 的 ``forward`` 用**闭包**包装（捕获 ``orig`` 这个已绑定的原方法），
  绝不依赖 Python 的 ``self`` 自动绑定 —— 上一版把普通函数赋给实例属性导致
  ``self`` 变成了输入的 Tensor，就是这个坑。

License: MIT — 缓存思路来自 TeaCache (ali-vilab / ComfyUI-TeaCache, Apache-2.0)。
"""

import torch

_WRAP_TAG = "_krea2_accel_wrapped"
_ORIG_BLOCK = "_krea2_accel_orig_block_forward"
_ORIG_DIT = "_krea2_accel_orig_dit_forward"
_STATE = "_krea2_accel_state"
_CFG = "_krea2_accel_cfg"


# ---------------------------------------------------------------------------
# 运行时状态
# ---------------------------------------------------------------------------
class _CacheState:
    """一次采样过程的缓存状态。每次检测到新的采样过程会自动重置。"""

    def __init__(self):
        self.reset_run()
        self.finished_calls = 0
        self.finished_skips = 0
        self.finished_dists = []

    def reset_run(self):
        self.prev_mod = None      # 上一步 block 0 的调制输入
        self.accum = 0.0          # 累加的相对 L1 距离
        self.residual = None      # 整个 block stack 的残差 R
        self.initial = None       # 本步 block 0 的输入（用于算 R）
        self.skip_count = 0       # 连续跳过计数
        self.skipping = False     # 本步是否处于跳过态
        self.last_t = None        # 上一次调用的时间步
        self.t_max = 0.0          # 本次采样过程的最大时间步（自动标定）
        self.progress = 0.0       # 采样进度 0..1
        self.in_range = True      # 是否落在 start/end percent 区间内
        self.calls = 0
        self.skips = 0
        self.dist_log = []        # 每一步的相对 L1 距离（用于选阈值）


def _find_dit(model_patcher):
    """从 ComfyUI 的 MODEL (ModelPatcher) 里找到真正的 Krea 2 DiT。"""
    seen = []
    candidates = [model_patcher]
    base = getattr(model_patcher, "model", None)
    if base is not None:
        candidates.append(base)
        candidates.append(getattr(base, "model", None))
        candidates.append(getattr(base, "diffusion_model", None))
    for cand in candidates:
        if cand is None or any(cand is s for s in seen):
            continue
        seen.append(cand)
        if hasattr(cand, "blocks") and hasattr(cand, "txtfusion"):
            return cand
    return None


# ---------------------------------------------------------------------------
# block forward 包装
# ---------------------------------------------------------------------------
def _make_block_forward(block, index, total, state, cfg):
    """返回一个替换 ``block.forward`` 的普通函数（闭包，不依赖 self 绑定）。"""
    orig = block.forward            # 已绑定的原生 forward
    mod = block.mod                 # DoubleSharedModulation
    prenorm = block.prenorm         # RMSNorm
    is_last = index == total - 1

    def forward(x, vec, freqs, mask=None, timestep_zero_index=None, transformer_options={}):
        if not cfg.get("enable", True):
            return orig(x, vec, freqs, mask, timestep_zero_index, transformer_options)

        # ---------------- block 0：为整个 stack 做缓存决策 ----------------
        if index == 0:
            state.calls += 1
            do_calc = True

            if state.in_range:
                try:
                    with torch.no_grad():
                        prescale, preshift = mod(vec)[:2]
                        if timestep_zero_index is not None:
                            bs = x.shape[0]
                            prescale = prescale[:bs]
                            preshift = preshift[:bs]
                        m = (1.0 + prescale) * prenorm(x) + preshift

                    prev = state.prev_mod
                    if prev is not None and prev.shape == m.shape:
                        denom = float(prev.abs().mean()) + 1e-8
                        rel = float((m - prev).abs().mean()) / denom
                        state.dist_log.append(round(rel, 4))
                        coef = cfg.get("coefficients")
                        state.accum += float(coef(rel)) if coef is not None else rel
                        if state.accum < cfg["thresh"] and state.skip_count < cfg["max_skip"]:
                            do_calc = False
                        else:
                            state.accum = 0.0
                    state.prev_mod = m
                except Exception as e:  # 任何异常都退回"正常计算"，绝不打断出图
                    print(f"[Krea2-Accel] 缓存判据计算失败，本次强制计算: {e}")
                    do_calc = True
                    state.accum = 0.0
                    state.prev_mod = None
            else:
                state.prev_mod = None
                state.accum = 0.0

            # ---- 命中：复用残差，后面所有 block 走恒等 ----
            resid = state.residual
            if not do_calc and resid is not None and resid.shape == x.shape:
                state.skip_count += 1
                state.skips += 1
                state.skipping = True
                state.initial = None
                if is_last:
                    state.skipping = False
                return x + resid.to(device=x.device, dtype=x.dtype)

            # ---- 未命中：正常计算并刷新残差 ----
            state.skip_count = 0
            state.skipping = False
            state.initial = x
            out = orig(x, vec, freqs, mask, timestep_zero_index, transformer_options)
            if is_last:
                state.residual = _store(out - x, cfg)
                state.initial = None
            return out

        # ---------------- block 1..N-1 ----------------
        if state.skipping:
            if is_last:
                state.skipping = False
            return x

        out = orig(x, vec, freqs, mask, timestep_zero_index, transformer_options)
        if is_last:
            base = state.initial
            if base is not None and base.shape == out.shape:
                state.residual = _store(out - base, cfg)
            state.initial = None
        return out

    return forward


def _store(residual, cfg):
    dev = cfg.get("device")
    if dev:
        try:
            return residual.to(dev)
        except Exception:
            return residual
    return residual


# ---------------------------------------------------------------------------
# SingleStreamDiT._forward 包装（只用于记录采样进度）
# ---------------------------------------------------------------------------
def _make_dit_forward(orig, state, cfg):
    """``SingleStreamDiT.forward`` 通过 ``self._forward(...)`` 调用，不会自动补 self，
    所以这里是一个形如 ``(x, timesteps, context, ...)`` 的普通函数。"""

    def forward(x, timesteps, context, attention_mask=None, ref_latents=None,
                transformer_options={}, **kwargs):
        try:
            tv = None
            if isinstance(timesteps, torch.Tensor):
                tv = float(timesteps.detach().reshape(-1)[0].item())
            elif timesteps is not None:
                tv = float(timesteps)

            if tv is not None:
                if state.last_t is None or tv > state.last_t + 1e-4:
                    # 时间步回升 => 新的一次采样开始
                    if state.calls > 0:
                        state.finished_calls = state.calls
                        state.finished_skips = state.skips
                        state.finished_dists = list(state.dist_log)
                        print(
                            "[Krea2-Accel] 上次采样: "
                            f"{state.calls} 次模型前向，复用 {state.skips} 次 "
                            f"({100.0 * state.skips / max(state.calls, 1):.1f}%)"
                        )
                        if state.finished_dists:
                            ds = sorted(state.finished_dists)
                            med = ds[len(ds) // 2]
                            print(
                                "[Krea2-Accel]   逐步相对L1: "
                                + " ".join(f"{v:.3f}" for v in state.finished_dists)
                                + f"  | 最小 {ds[0]:.3f} / 中位 {med:.3f} / 最大 {ds[-1]:.3f}"
                                + f"  -> 想多跳就把 rel_l1_thresh 调到 {med * 2:.2f} 附近"
                            )
                    state.reset_run()
                    state.t_max = tv
                state.last_t = tv

            p = 0.0
            if tv is not None and state.t_max > 1e-8:
                p = min(1.0, max(0.0, 1.0 - tv / state.t_max))
            state.progress = p
            state.in_range = (cfg.get("start", 0.0) <= p <= cfg.get("end", 1.0))
        except Exception as e:
            print(f"[Krea2-Accel] 进度记录失败（不影响出图）: {e}")
            state.in_range = True

        return orig(x, timesteps, context, attention_mask, ref_latents,
                    transformer_options, **kwargs)

    return forward


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------
def apply_krea2_cache(model, enable=True, thresh=0.12, start_p=0.0, end_p=1.0,
                      max_skip=2, cache_device="default", coefficients=None):
    """给 Krea 2 的 MODEL 注入残差缓存。原地修改，返回同一个对象以便串联。"""
    dm = _find_dit(model)
    if dm is None:
        raise RuntimeError(
            "Krea2CachePatch: 在这个 MODEL 里找不到 Krea 2 的 SingleStreamDiT "
            "（缺少 blocks / txtfusion 属性）。请先用「UNet加载器」或 "
            "「Load Diffusion Model」载入 krea2 模型，再接本节点。"
        )

    blocks = list(dm.blocks)
    if len(blocks) == 0:
        raise RuntimeError("Krea2CachePatch: 该模型的 blocks 为空，无法加速。")
    for i, b in enumerate(blocks):
        if not (hasattr(b, "mod") and hasattr(b, "prenorm")):
            raise RuntimeError(
                f"Krea2CachePatch: 第 {i} 个 block 不是 SingleStreamBlock（缺 mod/prenorm），"
                "本节点只支持 Krea 2 单流 DiT。"
            )

    device = None if cache_device in ("default", "gpu", "", None) else cache_device

    # 配置用同一个 dict 对象，闭包里直接读，重复执行节点即刷新参数
    cfg = getattr(dm, _CFG, None)
    if cfg is None:
        cfg = {}
        setattr(dm, _CFG, cfg)
    cfg.update({
        "enable": bool(enable),
        "thresh": float(thresh),
        "start": float(start_p),
        "end": float(end_p),
        "max_skip": int(max_skip),
        "device": device,
        "coefficients": coefficients,
    })

    state = getattr(dm, _STATE, None)
    if state is None:
        state = _CacheState()
        setattr(dm, _STATE, state)

    if getattr(dm, _WRAP_TAG, False):
        return model  # 已经包过，只更新配置

    for i, b in enumerate(blocks):
        if not hasattr(b, _ORIG_BLOCK):
            setattr(b, _ORIG_BLOCK, b.forward)
        b.forward = _make_block_forward(b, i, len(blocks), state, cfg)

    if not hasattr(dm, _ORIG_DIT):
        setattr(dm, _ORIG_DIT, dm._forward)
    dm._forward = _make_dit_forward(getattr(dm, _ORIG_DIT), state, cfg)

    setattr(dm, _WRAP_TAG, True)
    return model


def remove_krea2_cache(model):
    """卸载缓存包装，恢复原生 forward。"""
    dm = _find_dit(model)
    if dm is None or not getattr(dm, _WRAP_TAG, False):
        return model
    for b in dm.blocks:
        orig = getattr(b, _ORIG_BLOCK, None)
        if orig is not None:
            b.forward = orig
            try:
                delattr(b, _ORIG_BLOCK)
            except Exception:
                pass
    orig = getattr(dm, _ORIG_DIT, None)
    if orig is not None:
        dm._forward = orig
        try:
            delattr(dm, _ORIG_DIT)
        except Exception:
            pass
    if hasattr(dm, _WRAP_TAG):
        try:
            delattr(dm, _WRAP_TAG)
        except Exception:
            pass
    return model


def get_krea2_cache_stats(model):
    """返回最近一次采样的统计文本（用于 Krea2 Cache Stats 节点）。"""
    dm = _find_dit(model)
    state = getattr(dm, _STATE, None) if dm is not None else None
    if state is None:
        return "Krea2 Cache: 未安装缓存（MODEL 上没有 Krea2CachePatch）。"
    calls = state.finished_calls or state.calls
    skips = state.finished_skips or state.skips
    if getattr(dm, _WRAP_TAG, False) is False:
        return "Krea2 Cache: 未安装缓存（MODEL 上没有 Krea2CachePatch）。"
    cfg = getattr(dm, _CFG, {})
    ratio = 100.0 * skips / max(calls, 1)
    speed = calls / max(calls - skips, 1)
    dists = getattr(state, "finished_dists", None) or []
    if dists:
        ds = sorted(dists)
        med = ds[len(ds) // 2]
        dist_txt = (f"  逐步相对L1: {' '.join(f'{v:.3f}' for v in dists)}\n"
                    f"  最小 {ds[0]:.3f} / 中位 {med:.3f} / 最大 {ds[-1]:.3f}\n"
                    f"  建议阈值: 想多跳就往 {med * 2:.2f} 试，画质优先就 {med:.2f}\n")
    else:
        dist_txt = "  逐步相对L1: 暂无（复用 0 次时也可能没记到，请再跑一次）\n"
    return (
        f"Krea2 Cache (TeaCache)\n"
        f"  状态: {'已启用' if cfg.get('enable', True) else '已禁用'}\n"
        f"  模型前向次数: {calls}\n"
        f"  复用(跳过)次数: {skips}  ({ratio:.1f}%)\n"
        f"  理论加速: {speed:.2f}x\n"
        f"  阈值 rel_l1_thresh: {cfg.get('thresh', 0.0):.3f}\n"
        f"  生效区间: {cfg.get('start', 0.0):.2f} ~ {cfg.get('end', 1.0):.2f}\n"
        f"  连续跳过上限: {cfg.get('max_skip', 0)}\n"
        f"{dist_txt}"
        f"  (统计在下一次采样开始时才结算，请再跑一次看上一次的结果)"
    )
