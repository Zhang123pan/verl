from pathlib import Path

import collections

import pytest

from v35_online_cooperative_grpo.city_scheduler import CitySampler, SamplingConfig, SeedCursor, load_sampling_config


def test_seed_cursor_cycles_without_exhausting():
    cursor = SeedCursor((7, 8))
    assert [cursor.next() for _ in range(5)] == [7, 8, 7, 8, 7]


def test_yaml_city_sampler_uses_fixed_city_composition():
    root = Path(__file__).parents[2]
    cfg = load_sampling_config(root / "v35_online_cooperative_grpo/city_sampling.yaml")
    batch = CitySampler(cfg, seed=1).next_batch_cities()
    assert len(batch) == cfg.group_batch_size
    assert collections.Counter(batch) == {"jinan": 1, "hangzhou": 1, "newyork": 2}
    assert cfg.episode_seconds == 3600


def test_batch_size_must_match_city_units():
    cfg = SamplingConfig(3600, 30, 6, 6, None, load_sampling_config(Path(__file__).parents[2] / "v35_online_cooperative_grpo/city_sampling.yaml").cities)
    with pytest.raises(ValueError, match="multiple"):
        CitySampler(cfg)
