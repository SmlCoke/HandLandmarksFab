from __future__ import annotations

from collections.abc import Iterable, Sized
from typing import TypeVar

from tqdm.auto import tqdm


T = TypeVar("T")


def track_progress(
    items: Iterable[T],
    *,
    enabled: bool,
    description: str,
    unit: str,
) -> Iterable[T]:
    """Wrap an iterable in a terminal progress bar when requested."""
    if not enabled:
        return items
    total = len(items) if isinstance(items, Sized) else None
    return tqdm(
        items,
        total=total,
        desc=description,
        unit=unit,
        dynamic_ncols=True,
    )
