from v35_online_cooperative_grpo.sumo_adapter import SUMOEnvAdapter


class Inter:
    def __init__(self):
        self.inter_id = "i"
        self.control_phases = ["ETWT", "NTST", "ELWL", "NLSL"]
        self.dic_feature = {"traffic_movement_vehicle_ids_150m": [[], ["w"], [], [], ["e", "e2"], [], [], [], [], [], [], []]}
        self.dic_vehicle_speed_current_step = {"w": 0.0, "e": 2.0, "e2": 2.0}


class Env:
    def __init__(self): self.list_intersection=[Inter()]; self.time=0
    def reset(self, **_): pass
    def close(self): pass
    def snapshot(self, path): open(path,'w').write('state'); return path
    def load_from_file(self, *_, **__): pass
    def get_current_time(self): return self.time
    def step(self, actions, min_action_time): self.time+=min_action_time; return [],False,{'actions':actions}


def test_adapter_maps_phase_labels_and_runs_full_action_table(tmp_path):
    adapter=SUMOEnvAdapter(Env(), 1)
    assert adapter.background_actions()=={'i':'ETWT'}
    outcome=adapter.execute({'i':'NTST'},30)
    assert outcome['info']['actions']=={'i':1}
    assert outcome['endpoint']['i']['remaining_v_150m']==3.0


def test_cooperative_reward_averages_sender_and_receiver_costs():
    adapter = SUMOEnvAdapter(Env(), 1)
    result = {
        "endpoint": {
            "sender": {"queue_150m": 4.0, "remaining_v_150m": 10.0},
            "receiver": {"queue_150m": 2.0, "remaining_v_150m": 20.0},
        }
    }
    reward = adapter.compute_reward(
        {"focal_id": "sender", "receiver_ids": ("receiver",)}, result
    )
    assert reward == -4.5


def test_cooperative_reward_remains_backward_compatible_with_focal_only():
    adapter = SUMOEnvAdapter(Env(), 1)
    result = {"endpoint": {"sender": {"queue_150m": 4.0, "remaining_v_150m": 10.0}}}
    assert adapter.compute_reward({"focal_id": "sender"}, result) == -5.0
