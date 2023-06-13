import random
from typing import Any, Dict, List

import pytest

import infinity_core.api as api
from infinity_core.session import Session


def _default_session_without_api_call() -> Session:
    sesh = object.__new__(Session)
    sesh.token = "test-token"
    sesh.generator = "visionfit"
    sesh.server = api.DEFAULT_SERVER
    sesh._generator_info = {
        "name": "test-generator-v0.1.0",
        "version": "0.1.0",
        "params": [
            {
                "name": "param1",
                "type": "int",
                "default_value": 1,
                "options": {
                    "min": 1,
                    "max": 5,
                },
            },
            {"name": "param2", "type": "float", "default_value": 2.5, "options": {"min": -5.0, "max": 5.0}},
            {
                "name": "param3",
                "type": "str",
                "default_value": "CHOICE_1",
                "options": {"choices": ["CHOICE_1", "CHOICE_2"]},
            },
        ],
        "options": {
            "preview": True,
        },
    }

    return sesh


@pytest.fixture
def session() -> Session:
    return _default_session_without_api_call()


class TestSessionJobParamValidation:
    def test_valid_job_params(self, session: Session) -> None:
        job_params: List[Dict[str, Any]] = [{"param1": 2}, {"param2": 1.2}, {"param3": "CHOICE_2"}]
        assert all([v is None for v in session.validate_job_params(job_params=job_params)])

    def test_invalid_type(self, session: Session) -> None:
        job_params: List[Dict[str, Any]] = [{"param1": "one"}, {"param2": "two"}, {"param3": 1}]
        errors = session.validate_job_params(job_params=job_params)
        assert all([e is not None for e in errors])

    def test_casting_numerical_types(self, session: Session) -> None:
        job_params: List[Dict[str, Any]] = [{"param1": 1.0, "param2": 2}]
        assert all([v is None for v in session.validate_job_params(job_params=job_params)])

    def test_invalid_parameter(self, session: Session) -> None:
        job_params: List[Dict[str, Any]] = [{"param4": 1.5}]
        errors = session.validate_job_params(job_params=job_params)
        assert errors[0] is not None

    def test_out_of_range_parameter(self, session: Session) -> None:
        job_params: List[Dict[str, Any]] = [{"param1": 10}, {"param2": -100.0}]
        errors = session.validate_job_params(job_params=job_params)
        assert all([e is not None for e in errors])

    def test_out_of_choice_set(self, session: Session) -> None:
        job_params = [{"param3": "CHOICE_3"}]
        errors = session.validate_job_params(job_params=job_params)
        assert errors[0] is not None


class TestSessionSampling:
    @pytest.mark.parametrize("seed", [1, 100, 250, 475, 999])
    def test_random_sampling_ok_paths(self, seed: int, session: Session) -> None:
        random.seed(seed)
        random_job = session.random_job()
        p1_mn = session._generator_info["params"][0]["options"]["min"]
        p1_mx = session._generator_info["params"][0]["options"]["max"]
        p2_mn = session._generator_info["params"][1]["options"]["min"]
        p2_mx = session._generator_info["params"][1]["options"]["max"]
        p3_choices_list = session._generator_info["params"][2]["options"]["choices"]

        assert p1_mn <= random_job["param1"] <= p1_mx
        assert p2_mn <= random_job["param2"] <= p2_mx
        assert random_job["param3"] in p3_choices_list

    @pytest.mark.parametrize("seed", [2, 30, 175, 831, 1028])
    def test_default_value_fallback(self, seed: int, session: Session) -> None:
        session._generator_info["params"][0].pop("options")
        session._generator_info["params"][1]["options"].pop("min")
        session._generator_info["params"][2]["options"].pop("choices")
        random.seed(seed)
        random_job = session.random_job()
        p1_default_value = session._generator_info["params"][0]["default_value"]
        p2_default_value = session._generator_info["params"][1]["default_value"]
        p3_default_value = session._generator_info["params"][2]["default_value"]

        assert random_job["param1"] == p1_default_value
        assert random_job["param2"] == p2_default_value
        assert random_job["param3"] == p3_default_value
