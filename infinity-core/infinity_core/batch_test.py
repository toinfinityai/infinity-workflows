import pytest

import infinity_core.api as api
from infinity_core.batch import Batch, BatchSubmissionError, submit_batch
from infinity_core.data_structures import JobType


@pytest.fixture
def batch() -> Batch:
    return Batch(
        token="test-token",
        batch_id="123456789",
        name="test batch",
        jobs={"1": {"param1": 2.0}, "2": {"param2": True}},
        server=api.DEFAULT_SERVER,
        job_type=JobType.PREVIEW,
    )


class TestBatch:
    def test_job_ids_property(self, batch: Batch) -> None:
        assert set(batch.job_ids) == {"1", "2"}

    def test_job_params_property(self, batch: Batch) -> None:
        job_params = batch.job_params
        assert len(job_params) == 2

    def test_correct_num_successful_jobs(self, batch: Batch) -> None:
        assert batch.num_jobs == 2


class TestSubmitBatch:
    def test_reject_empty_string_token(self) -> None:
        token = ""
        with pytest.raises(BatchSubmissionError):
            submit_batch(
                token=token,
                generator="visionfit",
                job_type=JobType.PREVIEW,
                job_params=list({}),
                name="test batch",
                server=api.DEFAULT_SERVER,
            )

    def test_reject_empty_string_generator(self) -> None:
        generator = ""
        with pytest.raises(BatchSubmissionError):
            submit_batch(
                token="test-token",
                generator=generator,
                job_type=JobType.PREVIEW,
                job_params=list({}),
                name="test batch",
                server=api.DEFAULT_SERVER,
            )
