"""守候压缩包：一旦工作区里出现 zip/tar，立刻自动解压并自检。

用途：你从 Kaggle 下载完直接丢到 D:\\东大\\kaggle\\ 就行，不必再通知我，
本脚本发现文件后会立刻完成解压 + 自检。

用法:
    .\\run.ps1 tools\\watch_data.py --minutes 45
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WATCH_EXTS = (".zip", ".tar", ".gz", ".tgz", ".7z", ".rar")
SKIP_PARTS = {".venv", ".hf_cache", ".tmp", "runs", "submissions", "src", "tools", "_ref"}


def snapshot() -> dict[Path, float]:
    found: dict[Path, float] = {}
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_PARTS for part in p.parts):
            continue
        if p.name.lower().endswith(WATCH_EXTS):
            try:
                found[p] = p.stat().st_size
            except OSError:
                pass
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=45.0, help="最长等待分钟数")
    ap.add_argument("--interval", type=float, default=8.0, help="轮询间隔秒")
    args = ap.parse_args()

    print(f"[watch] 开始守候 {ROOT} 下的压缩包，最长 {args.minutes:g} 分钟，每 {args.interval:g}s 检查一次")
    before = snapshot()
    if before:
        print(f"[watch] 已有 {len(before)} 个压缩包，立即处理")

    deadline = time.time() + args.minutes * 60
    while time.time() < deadline:
        now = snapshot()
        # 新出现的文件，或体积已稳定（下载完成）的文件
        ready = [p for p, sz in now.items() if p not in before or sz != before.get(p)]
        if ready:
            # 等待体积稳定，避免解压到一半的包
            time.sleep(3)
            now2 = snapshot()
            stable = [p for p in ready if p in now2 and now2[p] == now.get(p)]
            if stable:
                print(f"[watch] 检测到 {len(stable)} 个压缩包，开始解压:")
                for p in stable:
                    print(f"        {p.name}  {p.stat().st_size/1024**2:.1f} MB")
                rc = subprocess.call(
                    [sys.executable, str(ROOT / "tools" / "ingest_data.py")],
                    cwd=str(ROOT),
                )
                print(f"[watch] ingest_data.py 退出码 = {rc}")
                return rc
            before = now
        time.sleep(args.interval)

    print(f"[watch] {args.minutes:g} 分钟内没有等到压缩包，退出")
    return 2


if __name__ == "__main__":
    sys.exit(main())
