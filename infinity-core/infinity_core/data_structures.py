""" Infinity API data structures.

This module contains common or important data structures used in other `infinity-core` modules.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Optional

JobParams = Dict[str, Any]


class HeaderKind(Enum):
    """Finite set of supported headers for HTTP requests."""

    AUTH = auto()
    JSON_CONTENT = auto()
    ACCEPT_JSON = auto()
    ACCEPT_OPENAPI_JSON = auto()
    ACCEPT_OPENAPI_YAML = auto()

    def to_header_dict(self, token: str) -> Dict[str, str]:
        """Convert header variant to header dictionary.

        Returns:
            Dict containing an HTTP request key-value pair.

        Raises:
            ValueError: If an unsupported header kind is provided.
        """
        if self == HeaderKind.AUTH:
            return {"Authorization": f"Token {token}"}
        elif self == HeaderKind.JSON_CONTENT:
            return {"Content-Type": "application/json"}
        elif self == HeaderKind.ACCEPT_JSON:
            return {"Accept": "application/json"}
        elif self == HeaderKind.ACCEPT_OPENAPI_JSON:
            return {"Accept": "application/vnd.oai.openapi+json"}
        elif self == HeaderKind.ACCEPT_OPENAPI_YAML:
            return {"Accept": "application/vnd.oai.openapi"}
        else:
            raise ValueError(f"Unsupported header kind {self}")


class JobType(Enum):
    """Fundamental job type supported by the Infinity API."""

    PREVIEW = auto()
    STANDARD = auto()


@dataclass(frozen=True)
class CompletedJob:
    """A data structured encapsulating a completed API job request.

    Args:
        job_id: Unique job ID.
        generator: Name of the generator for the job.
        params: Job parameters associated with the completed job.
        result_url: URL containing completed job result data, if available.
    """

    job_id: str
    generator: str
    params: JobParams
    result_url: Optional[str] = None

    def try_into_valid_completed_job(self) -> Optional["ValidCompletedJob"]:
        """Try and convert into a valid completed job.

        A valid completed job has a valid results URL from which generated synthetic data can be downloaded.

        Returns:
            Successfully constructed :obj:`ValidCompletedJob` or `None` if the job is invalid.
        """
        if self.result_url is not None:
            return ValidCompletedJob(
                job_id=self.job_id,
                generator=self.generator,
                params=self.params,
                result_url=self.result_url,
            )
        else:
            return None


@dataclass(frozen=True)
class ValidCompletedJob:
    """A data structured encapsulating a valid completed API job request.

    Args:
        job_id: Unique job ID.
        generator: Name of the generator for the job.
        params: Job parameters associated with the completed job.
        result_url: URL containing completed job result data.
    """

    job_id: str
    generator: str
    params: JobParams
    result_url: str

    @classmethod
    def try_from_completed_job(cls, completed_job: CompletedJob) -> Optional["ValidCompletedJob"]:
        """Try to create a valid completed job from a standard completed job.

        A valid completed job has a valid results URL from which generated synthetic data can be downloaded.

        Returns:
            Successfully constructed :obj:`ValidCompletedJob` or `None` if the job is invalid.
        """
        if completed_job.result_url is not None:
            return cls(
                job_id=completed_job.job_id,
                generator=completed_job.generator,
                params=completed_job.params,
                result_url=completed_job.result_url,
            )
        else:
            return None
