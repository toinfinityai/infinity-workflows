""" Infinity AI Session API for synthetic data generation.

This module provides a Session-style API to wrap interaction with the Infinity AI REST API. The
Session API abstracts away details of user authentication and enables ergonomics in areas such as
parameter validation and job parameter construction once a session is initialized.

Synthetic data generation requests (via `Session.submit`) return a `Batch` instance (detailed in
the `batch` module) which provides many facilities such as querying status of the batch, awaiting
full completion, and downloading ready results.
"""

import datetime
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import infinity_core.api as api
from infinity_core.batch import Batch, estimate_batch_samples, submit_batch
from infinity_core.data_structures import JobParams, JobType

_TYPE_STR_TO_ALLOWED_PYTHON_TYPE_SET: Dict[str, Set[Any]] = {
    "str": set([str]),
    "bool": set([bool]),
    "uuid": set([str, type(None)]),
}


_TYPE_STR_TO_CAST_TYPE: Dict[str, Any] = {"int": int, "float": float}


class ParameterValidationError(Exception):
    pass


class SessionInitializationError(Exception):
    pass


class BatchListRetrievalError(Exception):
    pass


class UsageStatsRetrievalError(Exception):
    pass


class AvailableGeneratorRetrievalError(Exception):
    pass


def get_available_generators(token: str, server: str) -> List[str]:
    """Get a list of available generators for the given token.

    Args:
        token: User authentication token.
        server: Base server URL.

    Returns:
        List of available generator names as strings.

    Raises:
        AvailableGeneratorRetrievalError: When the API request fails or fails to provide data.
    """
    try:
        r = api.get_all_generator_data(token=token, server=server)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise AvailableGeneratorRetrievalError(
            "Could not retrieve the list of generators available to the user from the API"
        ) from e

    return [generator["name"] for generator in data]


# TODO: Figure out how to get Sphinx to not document `_generator_info`.
@dataclass(frozen=False)
class Session:
    """An encapsulation of a user session to interact with the Infinity API.

    Args:
        token: Use authentication token.
        generator: Target generator for the session.
        server: URL of the target API server.
    """

    token: str
    generator: str
    server: str = api.DEFAULT_SERVER
    _generator_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            r = api.get_single_generator_data(token=self.token, generator_name=self.generator, server=self.server)
        except Exception as e:
            raise SessionInitializationError(
                "Failed to initialize session with generator information from the API"
            ) from e
        try:
            r.raise_for_status()
            generator_info = r.json()
        except Exception as e:
            if r.status_code == 401 or r.status_code == 403:
                raise SessionInitializationError("Invalid token") from e
            elif r.status_code == 500 or r.status_code == 404:
                raise SessionInitializationError(f"Invalid generator name {self.generator}") from e
            else:
                raise SessionInitializationError(
                    f"Failed to initialize session with generator information from the API with status code {r.status_code}"
                ) from e
        self._generator_info = generator_info

    def _validate_params(self, user_params: JobParams) -> Optional[str]:
        pinfo = self.parameter_info
        valid_parameter_set = set(pinfo.keys())
        unsupported_parameter_set = set()
        type_violation_list = list()
        type_cast_violation_list = list()
        constraint_violation_list = list()
        for uk, uv in user_params.items():
            if uk not in valid_parameter_set:
                unsupported_parameter_set.add(uk)
                continue
            expected_types = _TYPE_STR_TO_ALLOWED_PYTHON_TYPE_SET.get(pinfo[uk]["type"])
            if expected_types is not None:
                is_proper_type = any([isinstance(uv, ty) for ty in expected_types])
                if not is_proper_type:
                    type_violation_list.append((uk, type(uv), expected_types))
                    continue
            casting_type = _TYPE_STR_TO_CAST_TYPE.get(pinfo[uk]["type"])
            if casting_type is not None:
                try:
                    uv = casting_type(uv)
                except ValueError:
                    type_cast_violation_list.append((uk, casting_type))
                    continue
            param_options = pinfo[uk].get("options")
            if param_options is not None:
                if "min" in param_options:
                    cv = param_options["min"]
                    if uv < cv:
                        constraint_violation_list.append((uk, "min", cv, uv))
                if "max" in param_options:
                    cv = param_options["max"]
                    if uv > cv:
                        constraint_violation_list.append((uk, "max", cv, uv))
                if "choices" in param_options:
                    cv = param_options["choices"]
                    if uv not in cv:
                        constraint_violation_list.append((uk, "choices", cv, uv))

        had_unsupported_parameters = len(unsupported_parameter_set) > 0
        violated_types = len(type_violation_list) > 0
        violated_casting = len(type_cast_violation_list) > 0
        violated_constraints = len(constraint_violation_list) > 0

        if any(
            [
                had_unsupported_parameters,
                violated_types,
                violated_casting,
                violated_constraints,
            ]
        ):
            error_string = ""
            if had_unsupported_parameters:
                error_string += "\n\nUnsupported parameters:\n"
                unsupported_parameter_list = list(unsupported_parameter_set)
                for p in unsupported_parameter_list[0:-1]:
                    error_string += f"`{p}`, "
                error_string += f"`{unsupported_parameter_list[-1]}`"
            if violated_types:
                error_string += "\n\nType violations:\n"
                for p, a, e in type_violation_list:
                    error_string += f"Input parameter `{p}` expected one of compatible type(s) `{e}`, got `{a}`\n"
            if violated_casting:
                error_string += "\n\nType casting violations:\n"
                for p, c in type_cast_violation_list:
                    error_string += f"Input parameter `{p}` could not be cast as type `{c}` as expected\n"
            if violated_constraints:
                error_string += "\n\nConstraint violations:\n"
                for p, c, cv, pv in constraint_violation_list:
                    error_string += f"Input parameter `{p}` violated constraint `{c}` ({cv}) with value {pv}\n"
            return error_string
        else:
            return None

    def validate_job_params(self, job_params: List[JobParams]) -> List[Optional[str]]:
        """Check if a list of job parameters is valid.

        Args:
            job_params: A :obj:`list` of :obj:`dict`s containing job parameters for the batch.

        Returns:
            A list of validation errors (one per job param dict). Values will all be `None` if everything is valid.
        """
        return [self._validate_params(jp) for jp in job_params]

    # TODO: Make cached property that is compatible with 3.7+ and satisfies `mypy`.
    @property
    def parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """`dict`: Parameters of the generator with metadata."""
        param_info = dict()
        for p in self._generator_info["params"]:
            # TODO: Should we force-error if some of these are not provided?
            param_info[p["name"]] = {
                "type": p.get("type"),
                "default_value": p.get("default_value"),
                "options": p.get("options"),
            }

        return param_info

    # TODO: Make cached property that is compatible with 3.7+ and satisfies `mypy`.
    @property
    def default_job(self) -> JobParams:
        """:obj:`JobParams`: Default values for parameters of the generator."""
        return {k: d["default_value"] for k, d in self.parameter_info.items()}

    def random_job(self) -> JobParams:
        """Generate job parameters using uniform random sampling.

        This function will draw parameter values from a uniform distribution for all job parameters
        associated with a `min` and `max` constraint value or with a finite set of `choices`. For
        any parameters without these properties, the default value will be used.

        Returns:
            :obj:`JobParams` dictionary containing the randomly sampled parameters.
        """
        job_params: JobParams = dict()
        for k, v in self.parameter_info.items():
            if "options" in v.keys() and v["options"] is not None:
                if "choices" in v["options"]:
                    job_params[k] = random.choice(v["options"]["choices"])
                elif "min" in v["options"] and "max" in v["options"]:
                    mn, mx = v["options"]["min"], v["options"]["max"]
                    if v["type"] == "int":
                        job_params[k] = random.randint(mn, mx)
                    elif v["type"] == "float":
                        job_params[k] = random.uniform(mn, mx)
                    else:
                        job_params[k] = v["default_value"]
                else:
                    job_params[k] = v["default_value"]
            else:
                job_params[k] = v["default_value"]

        return job_params

    def randomize_unspecified_params(self, params: JobParams) -> JobParams:
        """Randomly populates missing parameters for a given job parameters dictionary

        Note that the sampling of missing parameters is uniform across the allowable range.
        Also note that this method operates on parameters for a single job, not a list of jobs.

        Args:
            params: A :obj:`dict` containing job parameters for a single job.

        Returns:
            A new job parameters :obj:`dict` with missing job parameters populated randomly.
        """
        return {**self.random_job(), **params}

    def submit(
        self,
        job_params: List[JobParams],
        is_preview: bool = False,
        batch_name: Optional[str] = None,
    ) -> Batch:
        """Submit a batch of 1 or more synthetic data jobs to the Infinity API.

        Note that any unspecified parameters for a given job parameter dictionary will take on their default values.

        Args:
            job_params: A :obj:`list` of :obj:`dict` containing job parameters for the batch.
            is_preview: Flag to indicate a preview is desired instead of a full job (e.g., video).
            batch_name: Optional descriptive for the submission.

        Returns:
            The created :obj:`Batch` instance.

        Raises:
            ValueError: If previews are requested but not allowed for the given generator.
            ParameterValidationError: If supplied or computed parameters are not supported by the generator.
        """
        # Check that previews are allowed for the given generator if a preview is requested.
        if is_preview and not self._generator_info.get("options", {}).get("preview", False):
            raise ValueError(f"Previews are not supported for `{self.generator}`")

        complete_params = self._set_default_params_and_validate(job_params)

        job_type = JobType.PREVIEW if is_preview else JobType.STANDARD
        batch = submit_batch(
            token=self.token,
            generator=self.generator,
            job_type=job_type,
            job_params=complete_params,
            name=batch_name,
            server=self.server,
        )
        return batch

    def estimate_samples(
        self,
        job_params: List[JobParams],
        is_preview: bool = False,
        batch_name: Optional[str] = None,
    ) -> List[int]:
        """Estimate number of samples or frames to be generated by the Infinity API for each job in a synthetic data
        Batch.

        Note that any unspecified parameters for a given job parameter dictionary will take on their default values.
        The accuracy of the estimate will vary by generator. For example, for some generators, the number of samples
        is deterministic, and the sample count estimate will be 100% accurate. For generators that introduce significant
        randomization, the accuracy will be lower. In general, aggregate accuracy of the estimate over the entire batch
        will increase with the number of jobs in the batch.

        Args:
            job_params: A :obj:`list` of :obj:`dict` containing job parameters for the batch.
            is_preview: Flag to indicate a preview is desired instead of a full job (e.g., video).
            batch_name: Optional descriptive for the submission.

        Returns:
            A :obj:`list` of sample estimates by job in the batch.

        Raises:
            ValueError: If previews are requested but not allowed for the given generator.
            ParameterValidationError: If supplied or computed parameters are not supported by the generator.
        """
        # Check that previews are allowed for the given generator if a preview is requested.
        if is_preview and not self._generator_info.get("options", {}).get("preview", False):
            raise ValueError(f"Previews are not supported for `{self.generator}`")

        complete_params = self._set_default_params_and_validate(job_params)

        job_type = JobType.PREVIEW if is_preview else JobType.STANDARD
        estimate = estimate_batch_samples(
            token=self.token,
            generator=self.generator,
            job_type=job_type,
            job_params=complete_params,
            name=batch_name,
            server=self.server,
        )
        return estimate

    def _set_default_params_and_validate(self, job_params: List[JobParams]) -> List[JobParams]:
        # Check just the user-supplied errors.
        user_input_errors = self.validate_job_params(job_params=job_params)
        if not all([v is None for v in user_input_errors]):
            error_str = "".join(
                [f"\n\njob param index {idx}: {e}" for idx, e in enumerate(user_input_errors) if e is not None]
            )
            raise ParameterValidationError(error_str)
        complete_params = [{**self.default_job, **jp} for jp in job_params]
        # Check total populated errors as well. TODO: Should we do this?
        errors = self.validate_job_params(job_params=complete_params)
        if not all([v is None for v in errors]):
            error_str = "".join([f"\n\njob param index {idx}: {e}" for idx, e in enumerate(errors) if e is not None])
            raise ParameterValidationError(error_str)
        return complete_params

    def batch_from_api(self, batch_id: str) -> Batch:
        """Reconstruct a previously submitted batch by unique ID.

        Args:
            batch_id: Unique batch ID of the target batch.

        Returns:
            A :obj:`Batch` instance for the target batch submission.
        """
        return Batch.from_api(token=self.token, batch_id=batch_id, server=self.server)

    def get_batches_last_n_days(self, n_days: int) -> List[Dict[str, Any]]:
        """Query the API for a list of batches submitted in the last N days.

        Args:
            n_days: Number of days looking back to gather submitted batches.

        Returns:
            A :obj:`list` containing batches and their metadata.

        Raises:
            BatchListRetrievalError: If the API request to get the list of batches fails.
        """
        end_time = datetime.datetime.now().astimezone()
        start_time = end_time - datetime.timedelta(days=n_days)
        try:
            r = api.get_batch_list(
                token=self.token,
                start_time=start_time,
                end_time=end_time,
                server=self.server,
            )
            r.raise_for_status()
            # Reverse the order of the list from the default provided by the API.
            data: List[Dict[str, Any]] = r.json()[::-1]
        except Exception as e:
            raise BatchListRetrievalError(
                f"Could not retrieve user batch list from the API for the last {n_days} days"
            ) from e
        for batch in data:
            batch["batch_id"] = batch.pop("id")
            batch["created"] = datetime.datetime.fromisoformat(batch["created"])

        return data

    def get_usage_stats_last_n_days(self, n_days: int) -> Dict[str, Any]:
        """Query the API for usage stats over the last N days.

        Args:
            n_days: Number of days looking back to gather usage stats for.

        Returns:
            A :obj:`dict` containing usage stats by generator.

        Raises:
            UsageStatsRetrievalError: If the API request to get usage stats fails.
        """
        try:
            r = api.get_usage_last_n_days(token=self.token, n_days=n_days, server=self.server)
            r.raise_for_status()
            data: Dict[str, Any] = r.json()
        except Exception as e:
            raise UsageStatsRetrievalError(
                f"Could not retrieve usage stats from the API for the last {n_days} days"
            ) from e

        return data
