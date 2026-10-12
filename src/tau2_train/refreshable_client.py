"""Wrapper around Tinker sampling client that auto-refreshes on auth failure."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class RefreshableSamplingClient:
    """Wraps a Tinker sampling client. On 401/auth errors, gets a fresh client
    from the training client and retries the call.

    All 320 concurrent rollouts share one instance. An asyncio.Lock ensures
    only one refresh happens at a time — other callers wait and reuse the
    new client.
    """

    def __init__(self, training_client: Any, sampling_client: Any):
        self._training_client = training_client
        self._client = sampling_client
        self._lock = asyncio.Lock()
        self._refresh_count = 0

    async def _refresh(self):
        async with self._lock:
            self._refresh_count += 1
            logger.warning(
                f"Refreshing sampling client (refresh #{self._refresh_count})"
            )
            self._client = (
                await self._training_client.save_weights_and_get_sampling_client_async()
            )

    async def sample_async(self, **kwargs):
        try:
            return await self._client.sample_async(**kwargs)
        except Exception as e:
            err_str = str(e)
            if "401" in err_str or "Invalid JWT" in err_str or "AuthenticationError" in type(e).__name__:
                await self._refresh()
                return await self._client.sample_async(**kwargs)
            raise

    @property
    def refresh_count(self) -> int:
        return self._refresh_count
