"""把一条命令脱离父进程启动（Windows），用于在本机跑长训练。

为什么需要
----------
本机两条路都不通：
1. DSH 的托管后台任务会把子进程的 CPU 份额压得很低，补丁增强线程被饿死，
   每个训练 step 从 0.44 s 掉到 7 s（实测 20 倍）。
2. `Start-Process` 新建的控制台会随父 pwsh 退出收到 CTRL_CLOSE，
   Python 直接 "program aborting due to window-CLOSE event" 挂掉。

所以这里用 DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP（可选 BREAKAWAY_FROM_JOB）
启动：没有控制台、不随父进程退出，stdout/stderr 重定向到文件。

用法
----
    .\\tools\\py.ps1 tools/spawn_detached.py --log .tmp/t4_train.log -- \\
        .venv\\Scripts\\python.exe -m src.vessel_train --tag v4_b0_384
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="stdout+stderr 重定向到的文件")
    ap.add_argument("--breakaway", action="store_true", help="尝试脱离父作业对象")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    cmd = [c for c in args.cmd if c != "--"]
    if not cmd:
        print("[X] 没给命令", file=sys.stderr)
        return 2

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    if args.breakaway:
        flags |= CREATE_BREAKAWAY_FROM_JOB

    with log_path.open("ab") as lf:
        try:
            p = subprocess.Popen(cmd, stdout=lf, stderr=lf, stdin=subprocess.DEVNULL,
                                 creationflags=flags, close_fds=True)
        except OSError as e:
            print(f"[!] 带 breakaway 启动失败（{e}），退回不带 breakaway", file=sys.stderr)
            flags &= ~CREATE_BREAKAWAY_FROM_JOB
            p = subprocess.Popen(cmd, stdout=lf, stderr=lf, stdin=subprocess.DEVNULL,
                                 creationflags=flags, close_fds=True)

    print(f"[spawn] pid={p.pid} 命令={' '.join(cmd)}")
    print(f"[spawn] 日志={log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
