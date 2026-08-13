from pathlib import Path

from v35_online_cooperative_grpo.city_env_config import build_city_env_configs


def test_three_city_configs_match_v30_network_files():
    root = Path(__file__).parents[4]
    configs, paths = build_city_env_configs(root)
    assert configs["jinan"]["ROADNET_FILE"] == "jinan_phase.net.xml"
    assert configs["hangzhou"]["NUM_INTERSECTIONS"] == 16
    assert configs["newyork"]["NUM_INTERSECTIONS"] == 196
    assert all(Path(value["PATH_TO_DATA"]).is_dir() for value in paths.values())
