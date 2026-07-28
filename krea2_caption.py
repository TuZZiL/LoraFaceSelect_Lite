from __future__ import annotations

import base64
import json
import os
import secrets
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LLAMA_SERVER = PROJECT_ROOT / "tools" / "llama.cpp" / "b10107" / "llama-server.exe"
DEFAULT_CAPTION_MODEL = (
    PROJECT_ROOT
    / "models"
    / "qwen3.5-caption"
    / "Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"
)
DEFAULT_CAPTION_MMPROJ = (
    PROJECT_ROOT
    / "models"
    / "qwen3.5-caption"
    / "mmproj-Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-f16.gguf"
)


class CaptionError(RuntimeError):
    pass


@dataclass(frozen=True)
class CaptionSettings:
    prompt: str
    server_path: Path = DEFAULT_LLAMA_SERVER
    model_path: Path = DEFAULT_CAPTION_MODEL
    mmproj_path: Path = DEFAULT_CAPTION_MMPROJ
    max_tokens: int = 160
    temperature: float = 0.2
    context_size: int = 4096


class LlamaCaptioner:
    def __init__(self, settings: CaptionSettings, log_path: Path) -> None:
        self.settings = settings
        self.log_path = log_path
        self._api_key = secrets.token_urlsafe(24)
        self._port = 0
        self._process: subprocess.Popen[bytes] | None = None
        self._log_handle: IO[bytes] | None = None

    def start(self) -> None:
        if self._process is not None:
            return
        self._validate_files()
        self._port = _free_local_port()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("wb")
        command = [
            str(self.settings.server_path),
            "-m",
            str(self.settings.model_path),
            "--mmproj",
            str(self.settings.mmproj_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(self._port),
            "-ngl",
            "99",
            "-c",
            str(self.settings.context_size),
            "-np",
            "1",
            "--api-key",
            self._api_key,
            "--no-webui",
            "--reasoning",
            "off",
        ]
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            self._process = subprocess.Popen(
                command,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
            self._wait_until_ready()
        except Exception:
            self.close()
            raise

    def caption_image(self, image_path: Path) -> str:
        if self._process is None or self._process.poll() is not None:
            raise CaptionError("Caption server is not running.")
        image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "model": "qwen3.5-caption",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                        {"type": "text", "text": self.settings.prompt},
                    ],
                }
            ],
            "max_tokens": self.settings.max_tokens,
            "temperature": self.settings.temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        response = self._request_json("/v1/chat/completions", payload, timeout=180)
        try:
            caption = response["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError) as exc:
            raise CaptionError(f"Unexpected caption response: {response}") from exc
        if not caption:
            raise CaptionError("Caption model returned an empty response.")
        return caption

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            except OSError:
                pass
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def _validate_files(self) -> None:
        for label, path in (
            ("llama-server", self.settings.server_path),
            ("caption model", self.settings.model_path),
            ("vision projector", self.settings.mmproj_path),
        ):
            if not path.is_file():
                raise CaptionError(f"{label} file does not exist: {path}")
        if not self.settings.prompt.strip():
            raise CaptionError("Caption prompt cannot be empty.")
        if not 1 <= self.settings.max_tokens <= 2048:
            raise CaptionError("Caption max tokens must be between 1 and 2048.")

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + 60
        last_error = "server did not respond"
        while time.monotonic() < deadline:
            assert self._process is not None
            if self._process.poll() is not None:
                raise CaptionError(
                    f"Caption server exited with code {self._process.returncode}: {self._log_tail()}"
                )
            try:
                response = self._request_json("/health", timeout=2)
                if response.get("status") == "ok":
                    return
                last_error = str(response)
            except CaptionError as exc:
                last_error = str(exc)
            time.sleep(0.25)
        raise CaptionError(f"Caption server was not ready after 60 seconds: {last_error}")

    def _request_json(
        self,
        route: str,
        payload: dict[str, object] | None = None,
        timeout: int = 10,
    ) -> dict[str, object]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            f"http://127.0.0.1:{self._port}{route}",
            data=data,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST" if data is not None else "GET",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise CaptionError(f"Caption API returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise CaptionError(f"Caption API request failed: {exc}") from exc
        if not isinstance(result, dict):
            raise CaptionError(f"Caption API returned invalid JSON: {result}")
        return result

    def _log_tail(self) -> str:
        if self._log_handle is not None:
            self._log_handle.flush()
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "server log unavailable"
        return text[-1500:].strip()


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
