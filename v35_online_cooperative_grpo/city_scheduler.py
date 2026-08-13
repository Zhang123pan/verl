"""City-balanced snapshot sampling and deterministic 3600-second seed rotation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CitySpec:
    name: str
    batch_units: int
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class SamplingConfig:
    episode_seconds: int
    control_period_seconds: int
    group_batch_size: int
    rollout_n: int
    sumo_branch_actor_limit: int | None
    cities: tuple[CitySpec, ...]


class SeedCursor:
    """Cycles deterministically through user-provided seeds forever."""

    def __init__(self, seeds: tuple[int, ...]):
        if not seeds:
            raise ValueError("Each enabled city requires at least one seed.")
        self.seeds = seeds
        self.index = 0

    def next(self) -> int:
        seed = self.seeds[self.index]
        self.index = (self.index + 1) % len(self.seeds)
        return seed


class CitySampler:
    """Builds each batch with a fixed, then shuffled, city composition."""

    def __init__(self, config: SamplingConfig, seed: int = 0):
        self.config = config
        self.random = random.Random(seed)
        self.cities = config.cities
        if not self.cities:
            raise ValueError("No enabled cities configured.")
        self.total_units = sum(city.batch_units for city in self.cities)
        if config.group_batch_size % self.total_units:
            raise ValueError(
                "group_batch_size must be a multiple of total city batch_units "
                f"({self.total_units}); got {config.group_batch_size}."
            )

    def next_batch_cities(self) -> list[str]:
        multiplier = self.config.group_batch_size // self.total_units
        cities = [city.name for city in self.cities for _ in range(city.batch_units * multiplier)]
        self.random.shuffle(cities)
        return cities


def load_sampling_config(path: str | Path) -> SamplingConfig:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load online city_sampling.yaml") from exc
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    episode_seconds = int(raw.get("episode_seconds", 3600))
    period = int(raw.get("control_period_seconds", 30))
    if episode_seconds <= 0 or period <= 0 or episode_seconds % period:
        raise ValueError("episode_seconds must be a positive multiple of control_period_seconds.")
    cities = []
    for name, item in (raw.get("cities") or {}).items():
        item = item or {}
        if not item.get("enabled", True):
            continue
        seeds = tuple(int(seed) for seed in item.get("seeds", ()))
        batch_units = int(item.get("batch_units", 1))
        if batch_units <= 0:
            raise ValueError(f"City {name!r} has non-positive batch_units.")
        cities.append(CitySpec(str(name), batch_units, seeds))
    return SamplingConfig(
        episode_seconds=episode_seconds,
        control_period_seconds=period,
        group_batch_size=int(raw.get("group_batch_size", 1)),
        rollout_n=int(raw.get("rollout_n", 6)),
        sumo_branch_actor_limit=raw.get("sumo_branch_actor_limit"),
        cities=tuple(cities),
    )
