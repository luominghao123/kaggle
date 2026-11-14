"""从 Kaggle 下载竞赛数据到工作区（沙箱内可用）。

为什么要自己写
--------------
本机 PowerShell / curl 的 TLS 不可用（schannel 报 SEC_E_NO_CREDENTIALS），
所以下载必须走 Python 自带的 OpenSSL；另外 DSH 沙箱只允许写工作区内路径，
所以这里默认落到 `.tmp/kaggle_dl/`，解压到 `data/task3/`。

凭证（三选一，按优先级）
------------------------
1. 环境变量 `KAGGLE_USERNAME` + `KAGGLE_KEY`（Basic 认证）
2. 环境变量 `KAGGLE_API_TOKEN` 或 `KGAT`（Bearer 认证，新版 token）
3. 工作区或 `~/.kaggle/kaggle.json`，内容形如 {"username":"...","key":"..."}

用法
----
    # 1) 先看有哪些文件 / 确认凭证有效
    .\\tools\\py.ps1 tools\\kaggle_fetch.py --slug 2026task3 --list

    # 2) 下载全部数据并解压到 data/task3
    .\\tools\\py.ps1 tools\\kaggle_fetch.py --slug 2026task3 --dest data/task3

    # 3) 只要某个文件
    .\\tools\\py.ps1 tools\\kaggle_fetch.py --slug 2026task3 --file sample_submission.csv
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://www.kaggle.com"
UA = "Mozilla/5.0 (kaggle_fetch)"


def load_auth() -> dict[str, str]:
    """返回请求头里的认证字段。"""
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    token = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KGAT")
    if token:
        return {"Authorization": f"Bearer {token}"}
    if user and key:
        raw = base64.b64encode(f"{user}:{key}".encode()).decode()
        return {"Authorization": f"Basic {raw}"}

    for cand in (ROOT / "kaggle.json", Path.home() / ".kaggle" / "kaggle.json"):
        if cand.exists():
            try:
                d = json.loads(cand.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if d.get("key") and d.get("username"):
                raw = base64.b64encode(f"{d['username']}:{d['key']}".encode()).decode()
                print(f"[auth] 使用 {cand}")
                return {"Authorization": f"Basic {raw}"}
            if d.get("token"):
                print(f"[auth] 使用 {cand} 里的 token")
                return {"Authorization": f"Bearer {d['token']}"}
    return {}


def api(path: str, auth: dict, *, method: str = "GET", body: dict | None = None,
        raw: bool = False):
    url = BASE + path
    headers = {"User-Agent": UA, "Accept": "application/octet-stream" if raw else "application/json"}
    headers.update(auth)
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=300, context=ctx) as r:
            payload = r.read()
            return payload if raw else payload.decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "replace")[:400]
        hint = ""
        if e.code == 401:
            hint = "\n  -> 没认证成功：请设置 KAGGLE_USERNAME/KAGGLE_KEY（或 KAGGLE_API_TOKEN），" \
                   "或把 kaggle.json 放到工作区根目录。"
        elif e.code == 403:
            hint = "\n  -> 403：多半是还没在竞赛页面点『Join / I Understand and Accept』同意规则。"
        elif e.code == 404:
            hint = "\n  -> 404：竞赛名写错，或该文件不存在。"
        raise RuntimeError(f"HTTP {e.code} {path}\n  {msg}{hint}") from None


def list_files(slug: str, auth: dict) -> list[dict]:
    txt = api(f"/api/v1/competitions/data/list/{slug}", auth)
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        print("[!] 返回内容不是 JSON：", txt[:300])
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="2026task3")
    ap.add_argument("--dest", default=None, help="解压目标目录（默认 data/<slug>）")
    ap.add_argument("--list", action="store_true", help="只列出文件")
    ap.add_argument("--file", default=None, help="只下载单个文件")
    ap.add_argument("--cache", default=".tmp/kaggle_dl", help="压缩包落地目录")
    args = ap.parse_args()

    auth = load_auth()
    if not auth:
        print("[X] 没有找到 Kaggle 凭证。三种办法任选其一：")
        print("    1) 设置环境变量：$env:KAGGLE_USERNAME='...'; $env:KAGGLE_KEY='...'")
        print("    2) 设置新版 token：$env:KAGGLE_API_TOKEN='KGAT_...'")
        print("    3) 把 kaggle.com/settings -> Create New Token 下到的 kaggle.json 放到工作区根目录")
        return 1
    print("[auth] OK")

    files = [] if args.file else list_files(args.slug, auth)
    if files:
        print(f"[list] {args.slug} 共 {len(files)} 个文件：")
        for f in files:
            mb = f.get("totalBytes", 0) / 1024**2
            print(f"    {f.get('name', '?'):50s} {mb:10.1f} MB")
    if args.list:
        return 0

    cache = (ROOT / args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    dest = Path(args.dest) if args.dest else ROOT / "data" / args.slug
    dest.mkdir(parents=True, exist_ok=True)

    if args.file:
        out = cache / args.file
        print(f"[down] {args.file} ...")
        blob = api(f"/api/v1/competitions/data/download/{args.slug}/{args.file}", auth, raw=True)
        out.write_bytes(blob)
        print(f"    -> {out}  {len(blob)/1024**2:.1f} MB")
        return 0

    zip_path = cache / f"{args.slug}.zip"
    print(f"[down] 下载整包（可能需要几分钟）...")
    blob = api(f"/api/v1/competitions/data/download-all/{args.slug}", auth, raw=True)
    zip_path.write_bytes(blob)
    print(f"    -> {zip_path}  {len(blob)/1024**2:.1f} MB")

    print(f"[unzip] -> {dest}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    for p in sorted(dest.rglob("*")):
        if p.is_file() and p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            print(f"    {p.relative_to(dest)}  {p.stat().st_size/1024:.0f} KB")

    print("\n下一步:")
    print(f"    .\\run.ps1 -m src.fovea_data {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
