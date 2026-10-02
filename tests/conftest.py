from __future__ import annotations

from pathlib import Path

import pytest

from acbench.arms import SdkSettings
from acbench.runner import RunConfig, run
from acbench.testing import FakeDecisionService


@pytest.fixture
def toy_run(tmp_path: Path):
    """Run the toy suite offline; returns (run_dir, decision service)."""

    def go(arms=("baseline", "sdk"), fault=None, trials=1, **overrides):
        service = FakeDecisionService(fault=fault)
        config = RunConfig(
            suite="toy", agent_model="fake/toy", arms=list(arms), trials=trials, out_dir=tmp_path,
            **{"workers": 4, **overrides},
        )
        return run(config, SdkSettings(http_client=service.client)), service

    return go
