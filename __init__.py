__version__ = "0.3.0"

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]

# 启动横幅：用来确认这个包**真的**加载了、加载的是哪个版本。
# ComfyUI 会静默吞掉自定义节点的导入异常，所以如果启动时看不到这一行，
# 说明包没导入成功（traceback 会打印在它上面）。
#
# v0.3.0 = 闭包式 block 包装版（已修复 "'Tensor' object has no attribute '_krea2_cfg'"）
# 如果你仍然看到 _krea2_cfg 报错，说明磁盘上跑的还是 v0.2.x 的旧文件。
_names = ", ".join(NODE_CLASS_MAPPINGS.keys())
print(f"[ComfyUI-Krea2-Accel] v{__version__} loaded OK — nodes: {_names}")
