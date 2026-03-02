from __future__ import annotations

import os
from typing import Literal

from openai import OpenAI

YC_API_KEY = (os.getenv("YC_API_KEY") or "").strip()
YC_FOLDER_ID = (os.getenv("YC_FOLDER_ID") or "").strip()

# Optional override, e.g. YC_TRANSLATE_MODEL="gpt://<folder>/yandexgpt-lite/latest"
YC_TRANSLATE_MODEL = (os.getenv("YC_TRANSLATE_MODEL") or "").strip()

_OPENAI_CLIENT: OpenAI | None = None


def _get_client() -> OpenAI:
    if not YC_API_KEY or not YC_FOLDER_ID:
        raise RuntimeError("YC_API_KEY and YC_FOLDER_ID must be set for YandexGPT translation")

    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        _OPENAI_CLIENT = OpenAI(
            api_key="DUMMY",
            base_url="https://llm.api.cloud.yandex.net/v1",
            default_headers={
                "Authorization": f"Api-Key {YC_API_KEY}",
                "OpenAI-Project": YC_FOLDER_ID,
            },
        )
    return _OPENAI_CLIENT


def _model_name() -> str:
    if YC_TRANSLATE_MODEL:
        return YC_TRANSLATE_MODEL
    return f"gpt://{YC_FOLDER_ID}/yandexgpt-lite/latest"


def translate(
    text: str,
    *,
    source_lang: Literal["ru", "en"],
    target_lang: Literal["ru", "en"],
) -> str:
    """Translate text between RU and EN using YandexGPT via OpenAI-compatible API.

    If source_lang == target_lang, the text is returned unchanged.
    """
    if not text:
        return text
    if source_lang == target_lang:
        return text

    client = _get_client()

    if source_lang == "ru" and target_lang == "en":
        system = "Translate the following text from Russian to English. Keep the meaning, no extra comments."
    elif source_lang == "en" and target_lang == "ru":
        system = "Переведи следующий текст с английского на русский. Сохрани смысл, без пояснений."
    else:
        raise ValueError(f"Unsupported translation direction: {source_lang}->{target_lang}")

    resp = client.chat.completions.create(
        model=_model_name(),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        temperature=0,
        max_tokens=2048,
    )

    content = resp.choices[0].message.content or ""
    return content.strip()