from __future__ import annotations

import dataclasses
import os
from functools import lru_cache
from typing import TYPE_CHECKING

from IPython.core.getipython import get_ipython


@dataclasses.dataclass
class Settings:
    token: str
    codestral_api_key: str
    provider: str  # "codestral" or "github"

    def reset(self):
        global settings
        self.from_env.cache_clear()
        settings = self.from_env()

    @staticmethod
    @lru_cache(maxsize=1)
    def from_env():
        return Settings(
            token=Settings.get_token(),
            codestral_api_key=Settings.get_codestral_key(),
            provider=Settings.get_provider(),
        )

    @staticmethod
    def get_token() -> str:
        if env_token := os.environ.get("GITHUB_COPILOT_ACCESS_TOKEN", ""):
            return env_token
        else:
            ip = get_ipython()
            assert ip is not None
            db = ip.db

            if TYPE_CHECKING:
                return "String"

            return db.get("github_copilot_access_token", "")

    @staticmethod
    def get_codestral_key() -> str:
        if env_key := os.environ.get("CODESTRAL_API_KEY", ""):
            return env_key
        else:
            ip = get_ipython()
            if ip is not None:
                db = ip.db
                return db.get("codestral_api_key", "")
            return ""

    @staticmethod
    def get_provider() -> str:
        if env_provider := os.environ.get("COPILOT_PROVIDER", ""):
            return env_provider.lower()
        else:
            ip = get_ipython()
            if ip is not None:
                db = ip.db
                return db.get("copilot_provider", "codestral")  # Default to codestral
            return "codestral"


settings = Settings.from_env()
