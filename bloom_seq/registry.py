"""Discover plugins via ``importlib.metadata`` entry points."""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Type

from bloom_seq.errors import PluginLoadError, PluginNotFoundError

GROUP_ALPHABETS = "bloom_seq.alphabets"
GROUP_BACKBONES = "bloom_seq.backbones"
GROUP_PATTERN_INDEXES = "bloom_seq.pattern_indexes"
GROUP_DATA_SOURCES = "bloom_seq.data_sources"
GROUP_PLAUSIBILITY = "bloom_seq.plausibility"


def _load_entry_point(ep) -> Any:
    try:
        return ep.load()
    except Exception as e:
        raise PluginLoadError(f"Failed to load plugin {ep.name}: {e}") from e


@lru_cache(maxsize=1)
def _entry_points_map(group: str) -> Dict[str, Any]:
    try:
        from importlib.metadata import entry_points
    except ImportError:
        return {}

    eps = entry_points()
    if hasattr(eps, "select"):
        selected = eps.select(group=group)
    else:
        selected = eps.get(group, [])
    return {ep.name: ep for ep in selected}


class PluginGroup:
    """Named plugins in one entry-point group."""

    def __init__(self, group: str):
        self.group = group

    def names(self) -> List[str]:
        return sorted(_entry_points_map(self.group).keys())

    def get(self, name: str) -> Any:
        m = _entry_points_map(self.group)
        if name not in m:
            raise PluginNotFoundError(
                f"No plugin {name!r} in group {self.group!r}. "
                f"Available: {', '.join(self.names()) or '(none)'}"
            )
        return _load_entry_point(m[name])

    def get_class(self, name: str) -> Type:
        obj = self.get(name)
        if not isinstance(obj, type):
            raise PluginLoadError(f"Plugin {name!r} is not a class: {obj!r}")
        return obj


alphabet_plugins = PluginGroup(GROUP_ALPHABETS)
backbones = PluginGroup(GROUP_BACKBONES)
pattern_indexes = PluginGroup(GROUP_PATTERN_INDEXES)
data_sources = PluginGroup(GROUP_DATA_SOURCES)
plausibility = PluginGroup(GROUP_PLAUSIBILITY)


def list_all_plugins() -> Dict[str, List[str]]:
    return {
        "alphabets": alphabet_plugins.names(),
        "backbones": backbones.names(),
        "pattern_indexes": pattern_indexes.names(),
        "data_sources": data_sources.names(),
        "plausibility": plausibility.names(),
    }
