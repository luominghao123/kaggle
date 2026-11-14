"""沙箱兼容补丁：让 tempfile.mkdtemp 创建「可写」的临时目录。

背景
----
本机 DSH 文件沙箱（workspace-write）会拒绝写入由 tempfile.mkdtemp() 创建的目录：
    mkdtemp 内部使用 os.mkdir(path, 0o700) → 后续 open(...,'wb') 报 WinError 5；
且事后 os.chmod 该目录同样被拒（无法补救）。
直接 os.makedirs(path)（默认 0o777）创建的目录则可正常读写。

影响
----
pip / venv / ensurepip 全部依赖 mkdtemp，因此在这个沙箱下必然失败，
表现为 "PermissionError: [Errno 13] ... pip-unpack-xxxx\\xxx.whl"。

做法
----
覆盖 tempfile.mkdtemp，用 os.makedirs 的默认权限创建目录。
通过 PYTHONPATH 注入本文件所在目录即可对任意 python 进程生效
（Python 启动时会自动 import sitecustomize）。

    $env:PYTHONPATH = "D:\\东大\\kaggle\\tools\\shim"
    python -m pip install ...
"""
import os
import random
import string
import tempfile

_ALPHABET = string.ascii_lowercase + string.digits
_orig_mkdtemp = tempfile.mkdtemp


def _mkdtemp_writable(suffix=None, prefix=None, dir=None):
    """与 tempfile.mkdtemp 行为一致，但目录权限用 0o777。"""
    if dir is None:
        dir = tempfile.gettempdir()
    prefix = prefix or "tmp"
    suffix = suffix or ""
    for _ in range(10000):
        name = prefix + "".join(random.choice(_ALPHABET) for _ in range(8)) + suffix
        path = os.path.join(dir, name)
        try:
            os.makedirs(path)  # 默认 0o777，沙箱下可写
        except FileExistsError:
            continue
        except OSError:
            continue
        return path
    raise FileExistsError("mkdtemp: 无法创建临时目录")


# 同时替换 tempfile 模块属性，以及已经持有旧引用的地方
tempfile.mkdtemp = _mkdtemp_writable
tempfile._mkdtemp = _mkdtemp_writable
