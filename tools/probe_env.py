"""探测：PyPI 上 torch 的 Windows 轮子信息 + 本机 GPU 情况。

用途：在正式安装前判断「PyPI 默认源给的 torch 是否自带 CUDA」。
在 Windows 上，PyPI 的 torch 轮子历史上是 CPU-only，
CUDA 版本需要走 https://download.pytorch.org/whl/cuXXX。
"""
import json
import platform
import sys
import urllib.request

UA = {"User-Agent": "Mozilla/5.0"}


def get_json(url, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    print("=" * 68)
    print("Python :", sys.version.split()[0], platform.machine())
    print("=" * 68)

    info = get_json("https://pypi.org/pypi/torch/json")
    latest = info["info"]["version"]
    print(f"\n[PyPI] torch 最新版本: {latest}")
    print("  该版本下 win_amd64 轮子:")
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    found = False
    for f in info["releases"][latest]:
        if "win_amd64" in f["filename"]:
            mark = "  <== 匹配本机解释器" if tag in f["filename"] else ""
            print(f"    {f['filename']}  {f['size'] / 1e6:7.1f} MB{mark}")
            found = True
    if not found:
        print("    (无 win_amd64 轮子)")

    # 对比 download.pytorch.org 的 CUDA 轮子体积，判断 PyPI 是否为 CPU 版
    print("\n[download.pytorch.org] 各 CUDA 通道的 torch 轮子（cu128 为例）:")
    try:
        html = urllib.request.urlopen(
            urllib.request.Request(
                "https://download.pytorch.org/whl/cu128/torch/", headers=UA
            ),
            timeout=40,
        ).read().decode("utf-8", "ignore")
        import re

        rows = re.findall(r'>([^<]*win_amd64\.whl)<[^>]*>.*?(\d{2,3}\.\d)</a>', html)
        if not rows:
            names = re.findall(r"([^<>\s]+win_amd64\.whl)", html)
            for n in names[:6]:
                print("   ", n)
        else:
            for n, sz in rows[:8]:
                print(f"    {n}  ~{sz} MB")
    except Exception as e:
        print("    探测失败:", type(e).__name__, e)

    print("\n[本机] nvidia-smi 摘要:")
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        print("   ", out.stdout.strip() or out.stderr.strip())
    except Exception as e:
        print("    失败:", e)


if __name__ == "__main__":
    main()
