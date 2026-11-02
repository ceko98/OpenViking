# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import asyncio
import base64
import json
import subprocess
import threading
import time
from unittest.mock import patch

import pytest

from openviking.models.embedder.local_embedders import (
    get_local_model_default_dimension,
    get_local_model_spec,
)
from openviking.models.model_error import ModelError, ModelErrorKind
from openviking.models.vlm.backends import claude_code_vlm
from openviking.models.vlm.backends.claude_code_vlm import ClaudeCodeVLM
from openviking.models.vlm.base import VLMFactory, VLMResponse
from openviking.models.vlm.registry import is_valid_provider


def _stdout(text="OK", **result_overrides):
    result = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": text,
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 3,
            "cache_creation_input_tokens": 2,
            "cache_read_input_tokens": 5,
            "output_tokens": 7,
        },
    }
    result.update(result_overrides)
    return "\n".join(json.dumps(e) for e in [{"type": "system", "subtype": "init"}, result])


def _completed(stdout, returncode=0):
    return subprocess.CompletedProcess(["claude"], returncode, stdout=stdout, stderr="")


@pytest.fixture(autouse=True)
def _reset_semaphore():
    claude_code_vlm._semaphore = None
    yield
    claude_code_vlm._semaphore = None


def _vlm(**extra):
    return ClaudeCodeVLM({"provider": "claude-code", "api_key": "unused", **extra})


def test_text_completion_args_env_and_stdin():
    vlm = _vlm()
    with patch("subprocess.run", return_value=_completed(_stdout("hello"))) as run:
        out = vlm.get_completion(
            messages=[
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
            ]
        )
    assert out == "hello"
    args = run.call_args.args[0]
    kwargs = run.call_args.kwargs
    assert args[:2] == ["claude", "-p"]
    assert "--safe-mode" in args
    assert "--bare" not in args
    assert args[args.index("--tools") + 1] == ""
    assert "--strict-mcp-config" in args
    assert args[args.index("--mcp-config") + 1] == '{"mcpServers":{}}'
    assert "--disable-slash-commands" in args
    assert "--no-session-persistence" in args
    assert args[args.index("--model") + 1] == "haiku"
    assert args[args.index("--effort") + 1] == "medium"
    assert args[args.index("--system-prompt") + 1] == "be brief"
    env = kwargs["env"]
    assert env["OPENVIKING_MEMORY_ENABLED"] == "0"
    assert env["OPENVIKING_AUTO_RECALL"] == "0"
    assert env["OPENVIKING_AUTO_CAPTURE"] == "0"
    assert env["OPENVIKING_NO_AUTO_INJECT"] == "1"
    assert env["OPENVIKING_RECALL_COMPRESS"] == "off"
    assert env["OPENVIKING_RECALL_REWRITE"] == "off"
    assert kwargs["timeout"] == claude_code_vlm.DEFAULT_TIMEOUT
    msg = json.loads(kwargs["input"])
    assert msg["message"]["content"] == [{"type": "text", "text": "hi"}]


def test_tools_empty_returns_vlm_response_and_ignores_tools():
    vlm = _vlm()
    with patch("subprocess.run", return_value=_completed(_stdout("final"))):
        resp = vlm.get_completion(messages=[{"role": "user", "content": "x"}], tools=[])
    assert isinstance(resp, VLMResponse)
    assert resp.content == "final"
    assert resp.tool_calls == []
    assert resp.finish_reason == "stop"
    assert resp.usage["prompt_tokens"] == 10
    assert resp.usage["completion_tokens"] == 7


def test_max_tokens_maps_to_length():
    vlm = _vlm()
    with patch("subprocess.run", return_value=_completed(_stdout("cut", stop_reason="max_tokens"))):
        resp = vlm.get_completion(messages=[{"role": "user", "content": "x"}], tools=[])
    assert resp.finish_reason == "length"


def test_vision_data_url_becomes_image_block():
    vlm = _vlm()
    png = base64.b64encode(b"\x89PNG fake").decode()
    with patch("subprocess.run", return_value=_completed(_stdout("Red"))) as run:
        out = vlm.get_vision_completion(prompt="color?", images=[b"\x89PNG fake"])
    assert out == "Red"
    blocks = json.loads(run.call_args.kwargs["input"])["message"]["content"]
    image = next(b for b in blocks if b["type"] == "image")
    assert image["source"]["type"] == "base64"
    assert image["source"]["media_type"].startswith("image/")
    assert image["source"]["data"] == png
    assert {"type": "text", "text": "color?"} in blocks


def test_multi_turn_is_role_tagged():
    _, msg = claude_code_vlm.build_request(
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
    )
    texts = [b["text"] for b in msg["message"]["content"]]
    assert texts == ["[user]", "a", "[assistant]", "b"]


def test_async_completion():
    vlm = _vlm()
    with patch("subprocess.run", return_value=_completed(_stdout("async ok"))):
        out = asyncio.run(vlm.get_completion_async(messages=[{"role": "user", "content": "x"}]))
    assert out == "async ok"


def test_rate_limit_error_is_classified():
    vlm = _vlm()
    stdout = _stdout("limit", is_error=True, api_error_status=429)
    with patch("subprocess.run", return_value=_completed(stdout)):
        with pytest.raises(ModelError) as exc:
            vlm.get_completion(messages=[{"role": "user", "content": "x"}])
    assert exc.value.kind == ModelErrorKind.RATE_LIMIT_ERROR


def test_rejected_rate_limit_event_is_classified():
    event = {"type": "rate_limit_event", "rate_limit_info": {"status": "rejected"}}
    stdout = json.dumps(event) + "\n" + _stdout("nope", is_error=True)
    with pytest.raises(ModelError) as exc:
        claude_code_vlm.parse_output(stdout)
    assert exc.value.kind == ModelErrorKind.RATE_LIMIT_ERROR


def test_timeout_is_reported():
    vlm = _vlm(timeout=1)
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("claude", 1)):
        with pytest.raises(ModelError, match="timed out"):
            vlm.get_completion(messages=[{"role": "user", "content": "x"}])


def test_no_output_error_includes_stderr():
    vlm = _vlm()
    proc = subprocess.CompletedProcess(["claude"], 1, stdout="", stderr="boom")
    with patch("subprocess.run", return_value=proc):
        with pytest.raises(ModelError, match="boom"):
            vlm.get_completion(messages=[{"role": "user", "content": "x"}])


def test_global_concurrency_cap():
    first = _vlm(extra_request_body={"max_concurrent": 2})
    second = _vlm()  # second instance shares the same process-wide cap
    active = 0
    peak = 0
    lock = threading.Lock()

    def fake_run(*_a, **_k):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return _completed(_stdout())

    messages = [{"role": "user", "content": "x"}]
    with patch("subprocess.run", side_effect=fake_run):
        threads = [
            threading.Thread(target=v.get_completion, kwargs={"messages": messages})
            for v in (first, second) * 3
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert peak == 2


def test_default_concurrency_is_two():
    assert _vlm()._max_concurrent() == 2


def test_factory_and_registry():
    assert is_valid_provider("claude-code")
    created = VLMFactory._create_provider({"provider": "claude-code", "api_key": "unused"})
    assert isinstance(created, ClaudeCodeVLM)
    assert created.model == "haiku"


def test_is_available_checks_binary():
    with patch("shutil.which", return_value=None):
        assert not _vlm(command="nope").is_available()
    with patch("shutil.which", return_value="/usr/bin/claude"):
        assert _vlm().is_available()


def test_bge_small_en_spec():
    spec = get_local_model_spec("bge-small-en-v1.5-f16")
    assert spec.dimension == 384
    assert spec.filename == "bge-small-en-v1.5-f16.gguf"
    assert spec.query_instruction.startswith("Represent this sentence")


def test_qwen3_embedding_spec():
    spec = get_local_model_spec("qwen3-embedding-0.6b-q8_0")
    assert spec.dimension == 1024
    assert spec.filename == "Qwen3-Embedding-0.6B-Q8_0.gguf"
    assert "Qwen/Qwen3-Embedding-0.6B-GGUF" in spec.download_url
    assert spec.query_instruction.startswith("Instruct: ")
    assert spec.query_instruction.endswith("\nQuery:")


def test_local_spec_selection_does_not_touch_network():
    for name, dim in (("bge-small-en-v1.5-f16", 384), ("qwen3-embedding-0.6b-q8_0", 1024)):
        assert get_local_model_default_dimension(name) == dim
