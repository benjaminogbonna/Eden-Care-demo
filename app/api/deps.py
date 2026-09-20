from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException

from app.core.config import Settings, get_settings


def settings_dep() -> Settings:
    return get_settings()


async def require_api_key(
    settings: Annotated[Settings, Depends(settings_dep)],
    x_api_key: Annotated[str | None, Header(description="Required when API_KEY is configured")] = None,
) -> None:
    if settings.api_key is None:
        return
    expected = settings.api_key.get_secret_value()
    if not x_api_key or not secrets.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")
