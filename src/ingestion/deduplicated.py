import hashlib
import re
import unicodedata

from src.config.config import CONFIG


class Deduplicator:

    def __init__(self, redis_client, config = CONFIG):
        self.redis = redis_client
        dedup_config = config.get("dedup")
        self.ttl = dedup_config.get("redis_ttl")
        self.enabled = dedup_config.get("enabled", False)

    def _normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\s+", " ", text.strip())
        return text

    def _sha256(self, text: str) -> str:
        return hashlib.sha256(self._normalize(text).encode("utf-8")).hexdigest()

    async def add_if_new(self, text: str) -> bool:
        sha = self._sha256(text)

        key = f"chunk:hash:{sha}"

        result = await self.redis.set(
            key,
            "1",
            nx=True,       
            ex=self.ttl,   
        )

        return result is True