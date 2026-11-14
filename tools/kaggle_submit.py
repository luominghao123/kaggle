"""把提交文件上传到 Kaggle 竞赛，并回读分数。

凭证与 tools/kaggle_fetch.py 相同（三选一）：
  1. 环境变量 KAGGLE_USERNAME + KAGGLE_KEY
  2. 环境变量 KAGGLE_API_TOKEN / KGAT
  3. 工作区根目录或 ~/.kaggle/kaggle.json

用法:
    # 只看这份文件会怎么被提交（不上传）
    .\\tools\\py.ps1 tools\\kaggle_submit.py --slug 2026task3 --file submissions/t3_ens.csv --dry-run

    # 真提交（description 会显示在榜单的提交历史里）
    .\\tools\\py.ps1 tools\\kaggle_submit.py --slug 2026task3 --file submissions/t3_ens.csv `
        --desc "fovea heatmap effnet_b0+convnext 5fold"

    # 查最近几次提交与分数
    .\\tools\\py.ps1 tools\\kaggle_submit.py --slug 2026task3 --list
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://www.kaggle.com"
UA = "Mozilla/5.0 (kaggle_submit)"


def load_auth() -> dict[str, str]:
    token = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KGAT")
    if token:
        return {"Authorization": f"Bearer {token}"}
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
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


def call(path: str, auth: dict, *, data: bytes | None = None,
         headers_extra: dict | None = None) -> tuple[int, str]:
    headers = {"User-Agent": UA, "Accept": "application/json"}
    headers.update(auth)
    if headers_extra:
        headers.update(headers_extra)
    req = urllib.request.Request(BASE + path, data=data, headers=headers,
                                 method="POST" if data is not None else "GET")
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=300, context=ctx) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        hint = {401: "\n  -> 认证失败：检查 token / kaggle.json",
                403: "\n  -> 403：先到竞赛页面点 Join 并同意规则",
                400: "\n  -> 400：多半是提交文件格式/行数不符合 sample_submission"}.get(e.code, "")
        return e.code, body[:600] + hint


def multipart(fields: dict[str, str], file_field: str, file_path: Path) -> tuple[bytes, str]:
    boundary = "----dsh" + uuid.uuid4().hex
    parts: list[bytes] = []
    for k, v in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    blob = file_path.read_bytes()
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; "
        f"filename=\"{file_path.name}\"\r\nContent-Type: application/octet-stream\r\n\r\n".encode()
        + blob + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def list_submissions(slug: str, auth: dict) -> None:
    code, body = call(f"/api/v1/competitions/submissions/list/{slug}", auth)
    if code != 200:
        print(f"[X] 查询失败 HTTP {code}: {body}")
        return
    try:
        items = json.loads(body)
    except json.JSONDecodeError:
        print(body[:500])
        return
    print(f"[list] {slug} 最近提交：")
    for it in items[:10]:
        print(f"    {it.get('date', '?')}  score={it.get('publicScore') or it.get('score')}"
              f"  status={it.get('status')}  desc={str(it.get('description'))[:60]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="2026task3")
    ap.add_argument("--file", default=None)
    ap.add_argument("--desc", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--wait", type=int, default=60, help="提交后等待多少秒再回读分数")
    args = ap.parse_args()

    auth = load_auth()
    if not auth:
        print("[X] 没有 Kaggle 凭证，无法提交。三选一：")
        print("    1) 把 kaggle.com/settings -> API -> Create New Token 下的 kaggle.json")
        print("       放到 D:\\东大\\kaggle\\kaggle.json")
        print("    2) 设置 $env:KAGGLE_USERNAME / $env:KAGGLE_KEY")
        print("    3) 设置 $env:KAGGLE_API_TOKEN='KGAT_...'")
        return 1

    if args.list or not args.file:
        list_submissions(args.slug, auth)
        return 0

    f = Path(args.file)
    if not f.is_absolute():
        f = ROOT / f
    if not f.exists():
        print(f"[X] 文件不存在: {f}")
        return 1
    head = f.read_text(encoding="utf-8-sig").splitlines()[:3]
    print(f"[file] {f}  {f.stat().st_size} bytes")
    for line in head:
        print(f"    {line}")
    if args.dry_run:
        print("[dry-run] 不上传")
        return 0

    fields = {"competitionName": args.slug, "submissionFileName": f.name}
    if args.desc:
        fields["submissionDescription"] = args.desc
    body, ctype = multipart(fields, "file", f)
    print(f"[submit] 上传 {f.name} -> {args.slug} ...")
    code, resp = call("/api/v1/competitions/submissions/submit", auth,
                      data=body, headers_extra={"Content-Type": ctype})
    print(f"[submit] HTTP {code}: {resp[:300]}")
    if code not in (200, 201):
        return 1

    if args.wait > 0:
        print(f"[wait] {args.wait}s 后回读分数 ...")
        time.sleep(args.wait)
        list_submissions(args.slug, auth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
