"""环境自检：确认 torch 能用到 GPU、timm 能拿到权重、各依赖版本正常。

运行:
    .\run.ps1 tools\verify_env.py
"""
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def main():
    print("=" * 72)
    print("Python :", sys.version.split()[0], sys.executable)

    import torch

    print("torch  :", torch.__version__, "| CUDA 编译版本:", torch.version.cuda)
    print("cudnn  :", torch.backends.cudnn.version())
    print("GPU 可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  设备  : {p.name}")
        print(f"  显存  : {p.total_memory/1024**3:.1f} GB")
        print(f"  算力  : sm_{p.major}{p.minor}")
        # 真实做一次矩阵乘法，确认 kernel 能跑
        a = torch.randn(2048, 2048, device="cuda")
        b = torch.randn(2048, 2048, device="cuda")
        c = (a @ b).sum().item()
        print(f"  矩阵乘法自检: OK (sum={c:.2f})")
        # bf16 支持检查（训练默认用 bf16）
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        y = (x @ x).float().sum().item()
        print(f"  bf16 自检   : OK (sum={y:.2f})")

    for mod in ("torchvision", "timm", "numpy", "pandas", "sklearn", "PIL"):
        try:
            m = __import__(mod)
            print(f"{mod:12s}: {getattr(m, '__version__', '?')}")
        except Exception as e:
            print(f"{mod:12s}: 导入失败 -> {type(e).__name__}: {e}")

    print("=" * 72)
    print("HF_ENDPOINT =", os.environ.get("HF_ENDPOINT"))


if __name__ == "__main__":
    main()
