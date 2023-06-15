import json
from dataclasses import dataclass
from os.path import abspath, dirname
from pathlib import Path
from typing import Optional

import pytest

import infinity_core.api as api
from infinity_core.batch import estimate_batch_samples, submit_batch
from infinity_core.data_structures import JobType

pytestmark = pytest.mark.needsapi
CUR_DIR = abspath(dirname(__file__))


@dataclass(frozen=True)
class TestConfig:
    generator_name: str
    server: str
    test_preview_batch_id: Optional[str]
    test_preview_job_id: Optional[str]
    test_standard_batch_id: Optional[str]
    test_standard_job_id: Optional[str]
    token: str

    @classmethod
    def from_json(cls, path: Path) -> "TestConfig":
        with open(path, "r") as f:
            cfg = json.load(f)

        return cls(
            generator_name=cfg["generator_name"],
            server=cfg["server"],
            test_preview_batch_id=cfg.get("test_preview_batch_id"),
            test_preview_job_id=cfg.get("test_preview_job_id"),
            test_standard_batch_id=cfg.get("test_standard_batch_id"),
            test_standard_job_id=cfg.get("test_standard_job_id"),
            token=cfg["token"],
        )


@pytest.fixture
def cfg() -> TestConfig:
    exe_dir = Path(CUR_DIR)
    return TestConfig.from_json(exe_dir / "config.json")


@pytest.mark.integration
@pytest.mark.apiget
class TestApiGetRequestIntegration:
    def test_get_all_preview_job_data(self, cfg: TestConfig) -> None:
        r = api.get_all_preview_job_data(token=cfg.token, server=cfg.server)

        assert r.ok

    def test_get_single_preview_job_data(self, cfg: TestConfig) -> None:
        assert cfg.test_preview_job_id is not None
        r = api.get_single_preview_job_data(token=cfg.token, preview_id=cfg.test_preview_job_id, server=cfg.server)

        assert r.ok

    def test_get_batch_preview_job_data(self, cfg: TestConfig) -> None:
        assert cfg.test_preview_batch_id is not None
        r = api.get_batch_data(token=cfg.token, batch_id=cfg.test_preview_batch_id, server=cfg.server)

        assert r.ok

    def test_get_all_standard_data(self, cfg: TestConfig) -> None:
        r = api.get_all_standard_job_data(token=cfg.token, server=cfg.server)

        assert r.ok

    def test_get_single_standard_job_data(self, cfg: TestConfig) -> None:
        assert cfg.test_standard_job_id is not None
        r = api.get_single_standard_job_data(
            token=cfg.token, standard_job_id=cfg.test_standard_job_id, server=cfg.server
        )

        assert r.ok

    def test_get_batch_standard_job_data(self, cfg: TestConfig) -> None:
        assert cfg.test_standard_batch_id is not None
        r = api.get_batch_data(token=cfg.token, batch_id=cfg.test_standard_batch_id, server=cfg.server)

        assert r.ok

    def test_get_all_generator_data(self, cfg: TestConfig) -> None:
        r = api.get_all_generator_data(token=cfg.token, server=cfg.server)

        assert r.ok

    def test_get_single_generator_data(self, cfg: TestConfig) -> None:
        r = api.get_single_generator_data(token=cfg.token, generator_name=cfg.generator_name, server=cfg.server)

        assert r.ok

    def test_get_usage_datetime_range(self, cfg: TestConfig) -> None:
        r = api.get_usage_datetime_range(token=cfg.token, server=cfg.server)

        assert r.ok

    def test_get_usage_last_n_days(self, cfg: TestConfig) -> None:
        r = api.get_usage_last_n_days(token=cfg.token, n_days=30, server=cfg.server)

        assert r.ok


@pytest.mark.integration
@pytest.mark.apipost
class TestApiPostRequestIntegration:
    def test_post_batch_preview(self, cfg: TestConfig) -> None:
        r = api.post_batch(
            token=cfg.token,
            generator=cfg.generator_name,
            name="test",
            job_params=[{}],
            is_preview=True,
            server=cfg.server,
        )

        assert r.ok

    def test_post_batch_standard(self, cfg: TestConfig) -> None:
        r = api.post_batch(
            token=cfg.token,
            generator=cfg.generator_name,
            name="test",
            job_params=[{}],
            is_preview=False,
            server=cfg.server,
        )

        assert r.ok

    def test_post_multi_job_batch(self, cfg: TestConfig) -> None:
        r = api.post_batch(
            token=cfg.token,
            generator=cfg.generator_name,
            name="test",
            job_params=[{}, {}],
            is_preview=True,
            server=cfg.server,
        )

        assert r.ok


@pytest.mark.integration
@pytest.mark.apipost
class TestApiSampleEstimateRequestIntegration:
    def test_estimate_batch_preview_request(self, cfg: TestConfig) -> None:
        r = api.estimate_batch_samples(
            token=cfg.token,
            generator=cfg.generator_name,
            name="test",
            job_params=[{}],
            is_preview=True,
            server=cfg.server,
        )

        assert r.ok

    def test_estimate_batch_standard_request(self, cfg: TestConfig) -> None:
        r = api.estimate_batch_samples(
            token=cfg.token,
            generator=cfg.generator_name,
            name="test",
            job_params=[{}, {}],
            is_preview=False,
            server=cfg.server,
        )

        assert r.ok


@pytest.mark.integration
@pytest.mark.batchpost
class TestBatchSubmissionIntegration:
    def test_preview_batch(self, cfg: TestConfig) -> None:
        batch = submit_batch(
            token=cfg.token,
            generator=cfg.generator_name,
            job_type=JobType.PREVIEW,
            job_params=[dict()],
            name="Infinity Core preview batch post integration test",
            server=cfg.server,
        )
        assert batch.num_jobs == 1
        valid_completed_jobs = batch.await_completion(timeout=5 * 60)
        assert len(valid_completed_jobs) == 1

    def test_standard_batch(self, cfg: TestConfig) -> None:
        batch = submit_batch(
            token=cfg.token,
            generator=cfg.generator_name,
            job_type=JobType.STANDARD,
            job_params=[dict()],
            name="Infinity Core standard batch post integration test",
            server=cfg.server,
        )
        assert batch.num_jobs == 1
        valid_completed_jobs = batch.await_completion(timeout=30 * 60)
        assert len(valid_completed_jobs) == 1


@pytest.mark.integration
@pytest.mark.batchpost
class TestBatchEstimationIntegration:
    def test_estimate_preview_batch(self, cfg: TestConfig) -> None:
        estimate = estimate_batch_samples(
            cfg.token,
            cfg.generator_name,
            JobType.PREVIEW,
            [{}, {}],
            "Test estimation",
            cfg.server,
        )
        assert estimate == [1, 1]

    def test_estimate_standard_batch(self, cfg: TestConfig) -> None:
        estimate = estimate_batch_samples(
            cfg.token,
            cfg.generator_name,
            JobType.STANDARD,
            [{}, {}],
            "Test estimation",
            cfg.server,
        )
        assert len(estimate) == 2
        assert estimate[0] >= 1
        assert estimate[1] >= 1
