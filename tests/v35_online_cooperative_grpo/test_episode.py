from pathlib import Path
import json

from v35_online_cooperative_grpo.episode import CooperativeEpisode, GeneratedResponse, GroupContext


def _response(signal, message=True):
    msg = '<message><to movement="ET">east</to><to movement="WT">west</to></message>' if message else ''
    return f'<perception>{{}}</perception><mode>fast</mode><signal>{signal}</signal>{msg}'


class Env:
    def __init__(self, parent=None): self.parent = parent or self
    def clone(self, snapshot_id, branch_id): return Env(self.parent)
    def base_messages(self, intersection_id):
        return [{"role":"user","content":"Critical format warning:\n\n" +
                "Your response must use exactly one of these valid layouts:\n\nFast mode:\n<perception>...</perception>\n<mode>fast</mode>\n<signal>...</signal>\n\nSlow mode:\n<perception>...</perception>\n<mode>slow</mode>\n<reasoning>...</reasoning>\n<signal>...</signal>\n\nDo not output any text before the first tag or after the last tag.\nDo not omit, rename, or reorder any tag required by the selected mode.\nDo not use markdown code fences.\n"}]
    def background_actions(self, snapshot_id): return {"intersection_1_1":"NTST","intersection_2_1":"NTST"}
    def execute_cycle(self, actions, seconds): return {"actions":actions,"seconds":seconds}
    def local_reward(self, focal_id, receiver_ids, result): return float(len(receiver_ids))


class Policy:
    def generate(self, messages, *, branch_id, role):
        return GeneratedResponse(_response("ETWT", message=role == "sender"), {"branch_id":branch_id,"role":role})


def test_group_clones_snapshot_and_shares_background_actions():
    root = Path(__file__).parents[2]
    routes = json.loads((root / "v35_online_cooperative_grpo/artifacts/movement_routes_jinan.json").read_text())
    env = Env()
    ctx = GroupContext("jinan", "step_1", "intersection_1_1", "A", routes, env.background_actions("step_1"), rollout_n=2)
    branches = CooperativeEpisode(env, Policy(), ctx).run_group()
    assert len(branches) == 2
    assert all(x.environment_result["seconds"] == 30 for x in branches)
    assert all(x.actions["intersection_1_1"] == "ETWT" for x in branches)
    assert all(len(x.trajectories) == 2 for x in branches)
