from pathlib import Path

import pytest

from v35_online_cooperative_grpo.observation import V30Observation


def test_v30_observation_validates_four_direction_media(tmp_path):
    videos = {}
    for direction in ("E", "W", "N", "S"):
        path = tmp_path / f"{direction}.mp4"
        path.write_bytes(b"video")
        videos[direction] = str(path)
    observation = V30Observation("intersection_1_1", 1, 0.0, 30.0, videos)
    observation.validate()


def test_v30_observation_rejects_wrong_window(tmp_path):
    videos = {}
    for direction in ("E", "W", "N", "S"):
        path = tmp_path / f"{direction}.mp4"
        path.write_bytes(b"video")
        videos[direction] = str(path)
    with pytest.raises(ValueError, match="control period"):
        V30Observation("i", 1, 0.0, 60.0, videos).validate()
