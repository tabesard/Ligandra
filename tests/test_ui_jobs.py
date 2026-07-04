"""UI background-job + caching mechanism, exercised in a real Streamlit runtime.

Uses Streamlit's headless ``AppTest`` harness (no browser).  Verifies that a
heavy step runs off the main thread (status goes running -> done), the result is
cached (an identical run is served instantly), and the home page renders.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from ligandra.ui.jobs import hash_key  # noqa: E402

_JOB_SCRIPT = """
import time
import streamlit as st
from ligandra.ui.jobs import run_job

def _work(x):
    time.sleep(0.15)
    return x * 2

status, payload = run_job("job1", "cache-key-A", _work, 21)
st.session_state["status"] = status
st.session_state["payload"] = payload
"""


def test_hash_key_is_order_independent_and_distinct():
    assert hash_key({"a": 1, "b": 2}) == hash_key({"b": 2, "a": 1})
    assert hash_key("x", 1) != hash_key("x", 2)


def test_job_runs_in_background_then_caches():
    at = AppTest.from_string(_JOB_SCRIPT)
    at.run()
    # first pass: submitted to the pool, not blocking the script
    assert at.session_state["status"] in ("running", "done")

    # poll across reruns until it completes
    for _ in range(60):
        if at.session_state["status"] in ("done", "cached"):
            break
        time.sleep(0.1)
        at.run()
    assert at.session_state["status"] in ("done", "cached")
    assert at.session_state["payload"] == 42

    # the next run for the same key is served from cache (instant)
    at.run()
    assert at.session_state["status"] == "cached"
    assert at.session_state["payload"] == 42


def test_job_surfaces_errors_without_crashing():
    script = """
import streamlit as st
from ligandra.ui.jobs import run_job

def _boom():
    raise ValueError("kaboom")

status, payload = run_job("job_err", "err-key", _boom)
st.session_state["status"] = status
st.session_state["payload"] = str(payload)
"""
    at = AppTest.from_string(script)
    at.run()
    for _ in range(60):
        if at.session_state["status"] == "error":
            break
        time.sleep(0.1)
        at.run()
    assert at.session_state["status"] == "error"
    assert "kaboom" in at.session_state["payload"]


def test_home_page_renders():
    # generous timeout: the first render cold-imports every plugin registry
    at = AppTest.from_file("ligandra/ui/app.py").run(timeout=60)
    assert not at.exception
    # the config-preview + workflow hint render
    assert any("Ligandra" in str(t.value) for t in at.title)
