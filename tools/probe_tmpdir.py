"""诊断：mkdtemp 创建的目录为何无法写入，以及 chmod 能否修复。

结论用于决定 pip 是否需要提权运行。
"""
import os
import stat
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(ROOT, ".tmp")
os.makedirs(TMP, exist_ok=True)


def try_write(where, label):
    p = os.path.join(where, "probe.txt")
    try:
        with open(p, "wb") as f:
            f.write(b"x" * 32)
        os.remove(p)
        print(f"    {label:28s} write OK")
        return True
    except Exception as e:
        print(f"    {label:28s} write FAIL  {type(e).__name__}: {e}")
        return False


print("[1] 手写 os.mkdir(0o700)")
d1 = os.path.join(TMP, "manual700")
os.makedirs(d1, exist_ok=True)
try_write(d1, "os.makedirs 默认")

print("\n[2] tempfile.mkdtemp (原样)")
d2 = tempfile.mkdtemp(dir=TMP, prefix="mkdtemp-")
print(f"    dir = {d2}")
try_write(d2, "mkdtemp 原样")

print("\n[3] tempfile.mkdtemp + chmod 0o777")
d3 = tempfile.mkdtemp(dir=TMP, prefix="mkdtempchmod-")
os.chmod(d3, 0o777)
try_write(d3, "mkdtemp + chmod777")

print("\n[4] tempfile.mkdtemp + chmod 0o700 显式")
d4 = tempfile.mkdtemp(dir=TMP, prefix="mkdtemp700-")
os.chmod(d4, 0o700)
try_write(d4, "mkdtemp + chmod700")

print("\n[5] 用 tempfile.TemporaryDirectory 写文件")
try:
    with tempfile.TemporaryDirectory(dir=TMP) as td:
        try_write(td, "TemporaryDirectory")
except Exception as e:
    print("    TemporaryDirectory 失败:", e)

print("\n[6] 查看目录 ACL / stat")
for label, d in (("mkdtemp原样", d2), ("chmod777", d3), ("manual700", d1)):
    try:
        st = os.stat(d)
        print(f"    {label:12s} mode={stat.filemode(st.st_mode)} "
              f"uid={st.st_uid} gid={st.st_gid}")
    except Exception as e:
        print(f"    {label:12s} stat 失败 {e}")
