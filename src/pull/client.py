# -*- coding: utf-8 -*-
"""金玥数据（dyyboard.ltd）客户端：登录、文件列表、日常更新文件下载。

基于开源 HTTP 库 requests 实现（开源架构）：
    - 登录：POST /GuFile/login（表单 username / password，返回 session cookie）
    - 文件列表：GET /GuFile/stock/files（返回 {"files": [{"filename": "YYYY-MM-DD.csv"}]}）
    - 下载：GET /GuFile/stock/download?filename=...&adjust=...（返回 CSV 字节流）
"""
from __future__ import annotations

import logging

import requests

LOGIN_URL = "/GuFile/login"
FILES_URL = "/GuFile/stock/files"
DOWNLOAD_URL = "/GuFile/stock/download"


class DyyClient:
    """金玥数据 HTTP 客户端。"""

    def __init__(self, base_url: str, account: str, password: str,
                 timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.account = account
        self.password = password
        self.timeout = timeout
        # requests.Session：自动携带 cookie（登录后的 session），开源 HTTP 会话
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/126.0 Safari/537.36"),
        })

    def login(self) -> None:
        """登录并保存 session cookie。"""
        resp = self.session.post(
            f"{self.base_url}{LOGIN_URL}",
            data={"username": self.account, "password": self.password},
            timeout=self.timeout,
        )
        try:
            data = resp.json()
        except ValueError:
            raise RuntimeError(f"登录响应非 JSON（HTTP {resp.status_code}）")
        if not data.get("success"):
            raise RuntimeError(f"登录失败: {data.get('message')}")
        logging.info("登录成功: %s（跳转 %s）", self.account, data.get("redirect"))

    def list_files(self) -> list[str]:
        """获取日常更新文件列表（最新在前），返回文件名列表如 ["2026-08-31.csv"]。"""
        resp = self.session.get(
            f"{self.base_url}{FILES_URL}", timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(f"获取文件列表失败: {data.get('message')}")
        files = [f.get("filename") for f in (data.get("files") or []) if f.get("filename")]
        logging.info("日常更新文件 %d 个（最新: %s）",
                     len(files), files[0] if files else "-")
        return files

    def download(self, filename: str, adjust: str) -> bytes:
        """下载指定日期的日常更新文件（adjust: fqr/qqr/hqr），返回 CSV 字节。"""
        resp = self.session.get(
            f"{self.base_url}{DOWNLOAD_URL}",
            params={"filename": filename, "adjust": adjust},
            timeout=120,
        )
        resp.raise_for_status()
        if "json" in (resp.headers.get("Content-Type") or ""):
            try:
                data = resp.json()
                if not data.get("success"):
                    raise RuntimeError(f"下载失败: {data.get('message')}")
            except ValueError:
                pass
        return resp.content
