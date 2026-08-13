import json

from v35_online_cooperative_grpo.vllm_policy import OpenAIChatPolicy


def test_policy_payload_and_response(monkeypatch):
    seen = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "<signal>ETWT</signal>"}}]}).encode()

    def fake_urlopen(request, timeout):
        seen.update({"url": request.full_url, "timeout": timeout,
                     "payload": json.loads(request.data.decode())})
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = OpenAIChatPolicy("http://localhost:8000/v1/chat/completions", "model", temperature=0).generate(
        [{"role": "user", "content": "x"}], branch_id=2, role="sender"
    )
    assert result.text == "<signal>ETWT</signal>"
    assert seen["payload"]["temperature"] == 0
    assert seen["payload"]["model"] == "model"
