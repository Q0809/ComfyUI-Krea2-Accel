# ComfyUI-Krea2-Accel

给**本地运行的 Krea 2**（`Comfy-Org/Krea-2`，ComfyUI 中架构标签为 `krea2`）做的**推理加速节点包**。

核心是一个 TeaCache 风格的**残差缓存**：相邻采样步之间，DiT 那 28 个 block 合起来的"总更新量"变化极小，于是可以把整段 block 的计算跳过、直接复用上一步的残差。不训练、不改权重、不换 attention 实现，**只改推理时的调用路径**。

> **⚠️ 前提：本包只加速"本地模型"**
> 用 `UNet加载器` / `Load Diffusion Model` 载入的 `.safetensors` 才能加速。
> Krea 2 的**云端 Partner Node**（调用 Krea 官方 API 的那种）不在本地跑 DiT，本包对它**完全无效**。

---

## 目录

- [节点一览](#节点一览)
- [效果实测](#效果实测)
- [安装](#安装)
- [快速开始](#快速开始)
- [参数详解](#参数详解)
- [调参：照着控制台做，不要盲猜](#调参照着控制台做不要盲猜)
- [工作原理（源码级）](#工作原理源码级)
- [与同类方案的关系](#与同类方案的关系)
- [故障排查](#故障排查)
- [FAQ](#faq)
- [已知限制](#已知限制)
- [许可证与致谢](#许可证与致谢)

---

## 节点一览

全部位于 ComfyUI 节点菜单的 `krea2/acceleration` 分类下。

| 节点 | 内部名 | 作用 |
|------|--------|------|
| **Krea2 Cache Patch (TeaCache)** | `Krea2CachePatch` | 主力节点。给 MODEL 注入残差缓存，典型 **1.3×–2×** 提速 |
| **Krea2 Cache Stats** | `Krea2CacheStats` | 输出命中率、复用次数、理论加速比、逐帧距离，**用来确认加速到底有没有生效** |
| **Krea2 Load Model (FP8)** | `Krea2LoadDiffusionModelFP8` | 可选。以 `fp8_e4m3fn` 等精度加载权重（12.9B 模型从 bf16 的 ~26 GB 压到 ~12 GB） |

FP8 加载器是**可选的**：它只是对 ComfyUI 原生加载器的薄封装，你也可以继续用原生的 `UNet加载器`。它导入失败不会影响另外两个节点。

---

## 效果实测

平台：**RTX 4060 Laptop 8GB**，模型：**Krea2 Turbo fp8**，工作流为三采放大。

| 配置 | 三采总耗时 | 相对基线 |
|------|-----------|---------|
| 关闭缓存 | ~165 s | 1.00× |
| 开启缓存（首次，含预热） | 101 s | 1.63× |
| 开启缓存（稳态） | **85 s** | **1.94×** |

工作流形态：

```
UNETLoader → Krea2CachePatch → LoraLoaderModelOnly
   → KSamplerAdvanced(7步) → LatentUpscaleBy(1.5×) → KSamplerAdvanced(6步)
   → UltimateSDUpscale(3步 × 2分块)
```

**为什么显存越紧张收益越大：** 8GB 显存放不下 12.5GB 的 Krea2，ComfyUI 一直在按层把权重来回倒腾。一旦决定跳过，那 28 个 block 的权重**一次都不用加载**——省掉的不只是算力，还有 PCIe 上的权重搬运。所以小显存机器上这个优化的价值，往往比大显存机器上更明显。

> 你的实际倍率取决于：步数、分辨率、显存压力、以及 `rel_l1_thresh` 调得多激进。**步数越多，可复用的步数通常也越多。**

---

## 安装

### 方式 A：手动（无需 pip）

把仓库放到 `ComfyUI/custom_nodes/` 下：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_NAME/ComfyUI-Krea2-Accel.git
```

最终结构：

```
ComfyUI/custom_nodes/ComfyUI-Krea2-Accel/
    __init__.py
    nodes.py
    krea2_cache.py
    krea2_fp8.py
```

### 方式 B：直接覆盖文件

把 4 个 `.py` 文件复制到 `custom_nodes/<任意目录名>/` 即可，**不需要 `pip install`**，无第三方依赖（只用 `torch` / `numpy`，ComfyUI 自带）。

### 验证安装

**必须重启 ComfyUI**。启动日志里应能看到：

```
[ComfyUI-Krea2-Accel] v0.3.0 loaded OK — nodes: Krea2CachePatch, Krea2CacheStats, ...
```

看不到这一行 = 包没导入成功，traceback 会打在它上面。

> ComfyUI 会**静默吞掉自定义节点的导入异常**，只提示"some nodes failed to load"，所以这条启动横幅是唯一的确认信号——这也是为什么要专门打这一行。

---

## 快速开始

**不要动你的模型加载器。** 保留原来的 `UNet加载器`（或 `Load Diffusion Model`），只把 **Krea2 Cache Patch** 串在它和采样器之间：

```
UNet加载器 (Krea2-MuseByStable_v30Turbo_fp8.safetensors)
      │  MODEL
      ▼
Krea2 Cache Patch ──MODEL──► KSampler / KSamplerAdvanced ──► VAEDecode ──► SaveImage
      │
      └──MODEL──► Krea2 Cache Stats   （可选，看命中率）
```

Turbo 模型推荐采样参数：**steps = 8，cfg = 1.0，sampler = euler，scheduler = simple**。

就是这么简单——一个节点，串上去，跑。剩下的只是调阈值。

---

## 参数详解

### Krea2 Cache Patch (TeaCache)

| 参数 | 默认 | 说明 |
|------|------|------|
| `model` | — | MODEL 输入 |
| `enable` | `true` | 关闭时**完全走原生 forward**，零加速也零改动，方便 A/B 对比 |
| `rel_l1_thresh` | `0.30` | **主旋钮**。越大跳得越多 → 越快，但画质风险越高 |
| `start_percent` / `end_percent` | `0.0` / `1.0` | 只在指定采样进度区间内启用缓存。想保护首尾画质就设 `0.1` / `0.9` |
| `max_skip_steps` | `3` | 连续复用步数上限。防止长时间不真算导致画面崩坏 |
| `cache_device` | `default` | `default` / `gpu` = 残差留在模型同设备；`cpu` = 残差存内存（省显存，略慢） |
| `coefficients` | 空 | 进阶：5 个逗号分隔的 4 次多项式系数，如 `-450,280,-45,3.2,-0.02`。**留空 = 原始相对 L1 模式（推荐）** |

### Krea2 Cache Stats

输出一段文本，直接看就行：

```
Krea2 Cache (TeaCache)
  状态: 已启用
  模型前向次数: 7
  复用(跳过)次数: 2  (28.6%)
  理论加速: 1.40x
  阈值 rel_l1_thresh: 0.300
  生效区间: 0.00 ~ 1.00
  连续跳过上限: 3
  逐步相对L1: 0.14 0.16 0.19 0.23 0.30 0.42
  最小 0.140 / 中位 0.230 / 最大 0.420
  建议阈值: 想多跳就往 0.46 试，画质优先就 0.23
```

> **注意：统计是在"下一次采样开始"时才结算上一次的结果。**
> 也就是说第一张图跑完时数字还是空的，**跑第二张才能看到第一张的数据**。
> 判据是"时间步回升 = 新一轮采样开始"。

---

## 调参：照着控制台做，不要盲猜

跑一张图之后，ComfyUI 控制台会打印：

```
[Krea2-Accel] 上次采样: 7 次模型前向，复用 2 次 (28.6%)
[Krea2-Accel]   逐步相对L1: 0.14 0.16 0.19 0.23 0.30 0.42  | 最小 0.14 / 中位 0.23 / 最大 0.42  -> 想多跳就把 rel_l1_thresh 调到 0.46 附近
```

**关键点：判据是累加的。**
距离 `d₁` 之后若 `d₁ < 阈值` 就跳一步；下一步累加 `d₁ + d₂` 再和阈值比较。所以**阈值 ≈ 你愿意接受的两三步距离之和**，而不是单步距离。

| 想要的效果 | 阈值取法 |
|-----------|---------|
| 保守（几乎无损） | 中位距离 × 1 |
| **均衡（推荐）** | 中位距离 × 2（控制台已经帮你算好了建议值） |
| 激进 | 中位距离 × 3，并把 `max_skip_steps` 提到 3~4 |

**诊断口诀：**

- **复用 = 0 次** → 阈值太小，直接往上加。
- **画质变差** → 往下调；或 `start_percent=0.1` / `end_percent=0.9`；或 `max_skip_steps` 降到 1。
- **加速不明显但复用率很高** → 说明瓶颈不在 DiT（可能在 VAE 解码、LoRA、放大节点、或权重搬运上）。

---

## 工作原理（源码级）

### Krea 2 的结构

Krea 2 是**单流 DiT**（`comfy.ldm.krea2.model.SingleStreamDiT`）：文本 token 与图像 token 拼成一条序列 `combined`，顺序通过 `self.blocks`（28 个 `SingleStreamBlock`，宽度 6144）：

```python
combined = torch.cat((context, img), dim=1)
for i, block in enumerate(self.blocks):
    combined = block(combined, tvec, freqs, None,
                     timestep_zero_index=..., transformer_options=...)
```

每个 block 都是**残差形式**（返回 `x + Δ`），这正好是缓存能成立的前提。

### 缓存做了什么

- **命中缓存**：block 0 直接返回 `x + R`（`R` = 上次算好的整个 block stack 的总残差），block 1..N-1 全部退化成恒等映射 → **28 个 block 的 attention + MLP 全部跳过**。
- **未命中**：正常跑完 28 个 block，在最后一个 block 处刷新 `R`。

### 命中判据

取 block 0 的 **AdaLN 调制输入**

```
m = (1 + prescale) * prenorm(x) + preshift
```

与**上一步**的 `m` 求**相对 L1 距离**，累加后与 `rel_l1_thresh` 比较——这就是 TeaCache 的做法。这个量在相邻步之间很稳定，且计算成本只有一次 RMSNorm + 一次标量统计，几乎免费。

### 为什么这样打补丁是安全的

这是本包最花心思的部分，也是 v0.3.0 存在的原因：

- **不重写 `_forward`**，不碰任何 tensor 的 shape——所有 shape 仍然由 ComfyUI 原生代码处理。
- `_forward` 只包一层"记录当前 timestep"的壳，用于推算采样进度（供 `start_percent` / `end_percent` 使用）。
- 每个 block 的 `forward` 用**闭包**包装，捕获**已绑定的原方法**（`orig = block.forward`），**绝不依赖 Python 的 `self` 自动绑定**。
  > 上一版把普通函数赋给实例属性，Python 不会再自动注入 `self`，结果 `self` 变成了输入的 Tensor，报 `'Tensor' object has no attribute '_krea2_cfg'`。这个坑在 v0.3.0 已经彻底修掉。
- 判据计算的**任何异常都会退回"正常计算"**，绝不打断出图——最坏情况只是这一张没加速。

---

## 与同类方案的关系

| 方案 | 思路 | 与本包 |
|------|------|--------|
| **本包（TeaCache 残差缓存）** | 跳过整段 block 计算 | — |
| [TeaCache](https://github.com/AliyunContainerService/TeaCache) | 同上，已适配多模型 | 本包的缓存思想来源（Apache-2.0） |
| **CacheDiT** | 类似思路的另一种实现 | 可叠加，但建议二选一 |
| **SageAttention**（KJNodes） | 换更快的 attention 实现 | **可叠加**，两者正交（一个省计算，一个加速计算） |
| `torch.compile` / Compile Model | 图编译 + kernel 融合 | 效果不确定，建议二选一 |

推荐组合：**本包 + SageAttention**，一个负责"少算"，一个负责"算得快"。

---

## 故障排查

| 现象 | 原因 / 处理 |
|------|------------|
| 启动日志没有 `[ComfyUI-Krea2-Accel] ... loaded OK` | 包没导入成功。看这一行**上面**的 traceback。最常见是文件没放全（4 个 `.py` 缺一不可） |
| 报错 `在这个 MODEL 里找不到 Krea 2 的 SingleStreamDiT` | 节点接到了非 krea2 模型，或还没接模型加载器。先接 `UNet加载器` 载入 krea2 权重 |
| 报错 `第 N 个 block 不是 SingleStreamBlock` | 模型不是单流 DiT，本包不支持 |
| 报错 `coefficients 无效` | `coefficients` 必须恰好 5 个逗号分隔数字。不需要就**留空** |
| 复用率 0% | `rel_l1_thresh` 太小。按控制台建议值往上调 |
| 画面糊 / 细节丢 | 阈值调小；或设 `start_percent=0.1`；或 `max_skip_steps=1` |
| 加速了但提速不如预期 | 瓶颈可能在 VAE / 放大 / 权重搬运，不在 DiT。看 `Krea2 Cache Stats` 的理论加速比确认 |
| 报错里出现 `_krea2_cfg` | 磁盘上跑的还是 v0.2.x 旧文件，请完整覆盖更新到 v0.3.0 |
| 显存更紧张了 | 把 `cache_device` 设为 `cpu`（残差存内存，会略慢） |

---

## FAQ

**Q：需要重新训练 / 转换模型吗？**
不需要。本包只在推理时改调用路径，权重一个字节都不动。

**Q：会不会改变出图结果？**
会有轻微差异——这是"训练无关、**轻微有损**"的推理加速。阈值调得越激进差异越大。关键出图请做视觉核对。

**Q：能和 LoRA 一起用吗？**
可以。实测工作流里就是 `Krea2CachePatch → LoraLoaderModelOnly` 这样串的。

**Q：步数很少（比如 4 步）还有用吗？**
有用但空间小——可复用的步数本来就少。Turbo 类模型建议 8 步左右。

**Q：为什么第一张图看不到统计？**
统计在检测到"时间步回升 = 新一轮采样"时才结算上一次。所以第二张图开始时才会打印第一张的数据。

**Q：为什么没有 Krea 2 专属的多项式系数？**
社区目前只有 qwen-image 等模型的拟合系数。默认走**原始相对 L1 模式**，稳健但需要你手动调阈值；进阶用户可以自己拟合后从 `coefficients` 传入。

---

## 已知限制

- 加速是**训练无关、轻微有损**的，关键出图请做视觉核对。
- 与 `torch.compile` / Compile Model 节点同时使用时效果不确定，建议二选一。
- 只支持 ComfyUI 中架构为 `krea2` 的**单流 DiT**；云端 API 节点不适用。
- 多段采样（如高清放大）会在每次新的采样段开始时重新标定进度。

---

## 更新日志

### v0.3.0

- **修复** `'Tensor' object has no attribute '_krea2_cfg'`：block forward 改为闭包包装并捕获已绑定的原方法，不再依赖 `self` 自动绑定。
- 启动横幅带版本号，便于确认实际加载的是哪一份文件。
- FP8 加载器改为从 ComfyUI 的**实时节点注册表**解析加载器，不再硬编码 `comfy_extras.nodes_model_loading` 路径（该模块在新版 ComfyUI 如 0.33.x 中已不存在）。

---

## 许可证与致谢

MIT（见 [LICENSE](LICENSE)）。

缓存思想改编自 [TeaCache](https://github.com/AliyunContainerService/TeaCache)（Apache-2.0，ali-vilab / ComfyUI-TeaCache），感谢原作者的工作。

Krea 2 模型版权归 Krea / Comfy-Org 所有，本仓库不包含任何权重文件。

---

*English version: [README_EN.md](README_EN.md)*
