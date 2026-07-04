"""Background-job + result-cache helpers so long steps don't freeze the UI.

Streamlit runs a page's script synchronously, so a multi-minute fetch / train /
generate call blocks the whole browser tab behind its spinner.  These helpers:

* run the heavy (pure) pipeline functions on a **background thread**, polling
  them across reruns so the tab stays responsive with a live elapsed timer; and
* **cache** each result keyed on its inputs, so re-running an identical
  target/model/generator is instant instead of recomputing.

The pipeline functions (``load_and_curate`` / ``train_and_benchmark`` /
``generate_and_rank``) are pure and never touch Streamlit, so they are safe to
run off the main thread.  Navigating away simply stops polling; the job finishes
in the background and its result is cached for when you return.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import streamlit as st


@st.cache_resource
def _executor() -> ThreadPoolExecutor:
    """One process-wide worker pool (survives reruns)."""
    return ThreadPoolExecutor(max_workers=2)


@st.cache_resource
def _result_cache() -> dict[str, Any]:
    """Process-wide result cache: input-hash -> result (frames/models/candidates)."""
    return {}


def hash_key(*parts: Any) -> str:
    """Stable short hash of arbitrary JSON-able inputs (a cache key)."""
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()[:16]


def clear_cache() -> None:
    """Drop all cached results (used by a 'recompute' control)."""
    _result_cache().clear()


def run_job(
    name: str, cache_key: str, fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> tuple[str, Any]:
    """Run ``fn(*args)`` in the background with caching; drive it across reruns.

    Returns ``(status, payload)``:

    * ``("cached", result)``  — served from cache (instant);
    * ``("done", result)``    — just finished (now cached too);
    * ``("running", elapsed)``— still working, ``elapsed`` seconds so far;
    * ``("error", exc)``      — raised; ``exc`` is the exception.
    """
    cache = _result_cache()
    if cache_key in cache:
        return "cached", cache[cache_key]

    jobs = st.session_state.setdefault("_jobs", {})
    job = jobs.get(name)
    # (re)start when unstarted or when the inputs (cache_key) changed
    if job is None or job["key"] != cache_key:
        jobs[name] = {
            "future": _executor().submit(fn, *args, **kwargs),
            "key": cache_key,
            "start": time.time(),
        }
        job = jobs[name]

    future: Future = job["future"]
    if not future.done():
        return "running", time.time() - job["start"]

    jobs.pop(name, None)
    exc = future.exception()
    if exc is not None:
        return "error", exc
    result = future.result()
    cache[cache_key] = result
    return "done", result


def render_job(
    name: str,
    cache_key: str,
    fn: Callable[..., Any],
    *args: Any,
    running_msg: str = "Working…",
    poll: float = 0.6,
    **kwargs: Any,
) -> tuple[str, Any]:
    """``run_job`` + a live status line; auto-reruns while the job is running.

    Pages call this and act on a ``("done"/"cached", result)`` return; the
    ``"running"`` state is handled here (status line + scheduled rerun).
    """
    status, payload = run_job(name, cache_key, fn, *args, **kwargs)
    if status == "running":
        st.info(f"⏳ {running_msg}  ·  {payload:.0f}s elapsed  (the tab stays responsive)")
        time.sleep(poll)
        st.rerun()
    return status, payload
