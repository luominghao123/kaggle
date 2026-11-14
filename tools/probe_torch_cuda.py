"""列出 download.pytorch.org 指定 CUDA 通道下、适配本机解释器的 torch 轮子。

用法: python tools/probe_torch_cuda.py [cu128]
"""
import re
import sys
import urllib.request

UA = {"User-Agent": "Mozilla/5.0"}


def fetch(url):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=UA), timeout=60
    ).read().decode("utf-8", "ignore")


def main():
    chan = sys.argv[1] if len(sys.argv) > 1 else "cu128"
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    print(f"通道 {chan} / 解释器 {tag} / win_amd64\n")

    for pkg in ("torch", "torchvision"):
        url = f"https://download.pytorch.org/whl/{chan}/{pkg}/"
        try:
            html = fetch(url)
        except Exception as e:
            print(f"[{pkg}] 抓取失败: {type(e).__name__} {e}")
            continue

        # 每个 <a ...>name</a> 后面跟一段 HTML，找到紧随其后的体积文本
        items = re.findall(r">([^<>]*win_amd64\.whl)</a>", html)
        items = [i for i in items if tag in i]
        print(f"[{pkg}] 共 {len(items)} 个匹配轮子，最新 6 个:")
        # 版本降序：从文件名里抽出版本号排序
        def ver(name):
            m = re.search(r"-(\d+\.\d+\.\d+)(?:\+[a-z0-9]+)?-", name)
            return tuple(int(x) for x in m.group(1).split(".")) if m else (0, 0, 0)

        for n in sorted(set(items), key=ver, reverse=True)[:6]:
            print("   ", n)
        print()


if __name__ == "__main__":
    main()
