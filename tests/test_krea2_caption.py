from __future__ import annotations

from pathlib import Path

from krea2_caption import CaptionSettings, LlamaCaptioner


class _RunningProcess:
    @staticmethod
    def poll() -> None:
        return None


def test_caption_request_contains_image_prompt_and_non_thinking_settings(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg bytes")
    captioner = LlamaCaptioner(CaptionSettings(prompt="Describe visible style."), tmp_path / "server.log")
    captioner._process = _RunningProcess()  # type: ignore[assignment]
    captured: dict[str, object] = {}

    def fake_request(route: str, payload: dict[str, object], timeout: int) -> dict[str, object]:
        captured.update({"route": route, "payload": payload, "timeout": timeout})
        return {"choices": [{"message": {"content": "  concise caption  "}}]}

    monkeypatch.setattr(captioner, "_request_json", fake_request)  # type: ignore[attr-defined]

    result = captioner.caption_image(image)

    assert result == "concise caption"
    assert captured["route"] == "/v1/chat/completions"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    messages = payload["messages"]
    assert isinstance(messages, list)
    content = messages[0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1] == {"type": "text", "text": "Describe visible style."}
