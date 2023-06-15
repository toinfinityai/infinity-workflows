""" Infinity AI synthetic data REST API wrapper.

This module provides lightweight Python wrapping of the Infinity AI API for synthetic data
generation. Use this module to directly interact with the Infinity API or to build higher level
abstractions for interfacing with the Infinity API. For example, the `batch` module provides a
higher level abstraction for batches of synthetic data and uses this module to interact directly
with the REST API.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlencode, urljoin

import requests
from requests.models import Response

from infinity_core.data_structures import HeaderKind

DEFAULT_SERVER: str = "https://api.toinfinity.ai"


class GetResponseRawBytesError(Exception):
    pass


class GetResponseJsonError(Exception):
    pass


class GetResponseTextError(Exception):
    pass


def _ensure_trailing_slash(url: str) -> str:
    return url if url.endswith("/") else url + "/"


def _ensure_no_trailing_slash(url: str) -> str:
    return url[:-1] if url.endswith("/") else url


def build_request(
    token: str,
    server: str,
    endpoint: str,
    headers: Optional[Set[HeaderKind]] = None,
    query_parameters: Optional[Dict[str, str]] = None,
) -> Tuple[str, Dict[str, str]]:
    """Builds HTTP requests.

    Args:
        token: User authentication token.
        server: Base server URL.
        endpoint: Target endpoint.
        headers: Optional set of request header types.
        query_parameters: Optional dictionary of query parameters.

    Returns:
        A tuple containing the target URL and a dictionary of headers.

    Raises:
        ValueError: `token`, `server`, or `endpoint` strings are empty.
    """
    if token == "":
        raise ValueError("`token` cannot be an empty string")
    if server == "":
        raise ValueError("`server` cannot be an empty string")
    if endpoint == "":
        raise ValueError("`endpoint` cannot be an empty string")

    url = urljoin(server, endpoint)
    url = _ensure_trailing_slash(url)
    if query_parameters is not None:
        url += "?" + urlencode(query_parameters)
        url = _ensure_no_trailing_slash(url)
    headers_dict: Dict[str, str] = dict()
    if headers is not None:
        for h in headers:
            headers_dict = {**headers_dict, **h.to_header_dict(token)}

    return url, headers_dict


def unwrap_raw_bytes_payload(response: Response) -> bytes:
    """Unwrap raw byte contents from provided HTTP request response.

    Args:
        response: HTTP request response.

    Returns:
        Raw byte payload returned from the request.

    Raises:
        GetResponseRawBytesError: When `response` has an error status code and/or has no `content` field.
    """
    try:
        response.raise_for_status()
        data = response.content
    except Exception as e:
        raise GetResponseRawBytesError("Failed attempting to unwrap raw byte content from the response") from e
    return data


def unwrap_json_payload(response: Response) -> Any:
    """Unwrap json payload from provided HTTP request response.

    Args:
        response: HTTP request response.

    Returns:
        Decoded JSON payload.

    Raises:
        GetResponseJsonError: When `response` has an error status code and/or has no `json()` content.
    """
    try:
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        raise GetResponseJsonError("Failed attempting to unwrap JSON content from the response") from e
    return data


def unwrap_text_payload(response: Response) -> str:
    """Unwrap contents from provided HTTP request response as unicode.

    Args:
        response: HTTP request response.

    Returns:
        Decoded JSON payload.

    Raises:
        GetResponseTextError: When `response` has an error status code and/or has no `test` field.
    """
    try:
        response.raise_for_status()
        data = response.text
    except Exception as e:
        raise GetResponseTextError("Failed attempting to unwrap text content from the response") from e
    return data


def get_all_preview_job_data(token: str, server: str = DEFAULT_SERVER) -> Response:
    """Get data for all previews associated with the given token.

    Args:
        token: User authentication token.
        server: Base server URL.

    Returns:
        HTTP request response.
    """
    url, headers = build_request(
        token=token,
        server=server,
        endpoint="api/job_previews/",
        headers=set([HeaderKind.AUTH, HeaderKind.ACCEPT_JSON]),
    )
    return requests.get(url=url, headers=headers)


def get_single_preview_job_data(token: str, preview_id: str, server: str = DEFAULT_SERVER) -> Response:
    """Get data for a given preview associated with the given token.

    Args:
        token: User authentication token.
        preview_id: Unique ID associated with a previously run preview.
        server: Base server URL.

    Returns:
        HTTP request response.
    """
    url, headers = build_request(
        token=token,
        server=server,
        endpoint=f"api/job_previews/{preview_id}/",
        headers=set([HeaderKind.AUTH, HeaderKind.ACCEPT_JSON]),
    )
    return requests.get(url=url, headers=headers)


def get_all_standard_job_data(token: str, server: str = DEFAULT_SERVER) -> Response:
    """Get data for all standard jobs associated with the given token.

    Args:
        token: User authentication token.
        server: Base server URL.

    Returns:
        HTTP request response.
    """
    url, headers = build_request(
        token=token,
        server=server,
        endpoint="api/job_runs/",
        headers=set([HeaderKind.AUTH, HeaderKind.ACCEPT_JSON]),
    )
    return requests.get(url=url, headers=headers)


def get_single_standard_job_data(token: str, standard_job_id: str, server: str = DEFAULT_SERVER) -> Response:
    """Get data for a given standard job associated with the given token.

    Args:
        token: User authentication token.
        standard_job_id: Unique ID associated with a previously run job.
        server: Base server URL.

    Returns:
        HTTP request response.
    """
    url, headers = build_request(
        token=token,
        server=server,
        endpoint=f"api/job_runs/{standard_job_id}/",
        headers=set([HeaderKind.AUTH, HeaderKind.ACCEPT_JSON]),
    )
    return requests.get(url=url, headers=headers)


def get_batch_list(
    token: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    server: str = DEFAULT_SERVER,
) -> Response:
    """Get a list of batches associated with the given token over some time range.

    Args:
        token: User authentication token.
        start_time: Optional start time for query.
        end_time: Optional end time for query.
        server: Base server URL.

    Returns:
        HTTP request response.

    Raises:
        ValueError: If `end_time` is chronologically before `start_time`.
    """
    if end_time is not None and start_time is not None:
        if end_time < start_time:
            raise ValueError(f"End time ({end_time}) before start time ({start_time}) for usage query")
    query_parameters = dict()
    if start_time is not None:
        if start_time.tzinfo is None:
            start_time = start_time.astimezone()
        query_parameters["start_time"] = start_time.isoformat()
    if end_time is not None:
        if end_time.tzinfo is None:
            end_time = end_time.astimezone()
        query_parameters["end_time"] = end_time.isoformat()
    url, headers = build_request(
        token=token,
        server=server,
        endpoint="api/batch/",
        headers=set([HeaderKind.AUTH, HeaderKind.ACCEPT_JSON]),
        query_parameters=query_parameters,
    )
    return requests.get(url=url, headers=headers)


def get_batch_data(token: str, batch_id: str, server: str) -> Response:
    """Get detailed information on a previously submitted batch.

    Args:
        token: User authentication token.
        batch_id: Unique ID associated with a previously submitted batch.
        server: Base server URL.

    Returns:
        HTTP request response.
    """

    headers_set = set([HeaderKind.AUTH, HeaderKind.ACCEPT_JSON])
    url, headers = build_request(
        token=token,
        server=server,
        endpoint=f"api/batch/{batch_id}/",
        headers=headers_set,
    )
    return requests.get(url=url, headers=headers)


def get_batch_summary_data(token: str, batch_id: str, server: str) -> Response:
    """Get detailed information on a previously submitted batch.

    Args:
        token: User authentication token.
        batch_id: Unique ID associated with a previously submitted batch.
        server: Base server URL.

    Returns:
        HTTP request response.
    """
    headers_set = set([HeaderKind.AUTH, HeaderKind.ACCEPT_JSON])
    url, headers = build_request(
        token=token,
        server=server,
        endpoint=f"api/batch/{batch_id}/summary/",
        headers=headers_set,
    )
    return requests.get(url=url, headers=headers)


def get_all_generator_data(token: str, server: str = DEFAULT_SERVER) -> Response:
    """Get data on all of the generators associated with the given token.

    Args:
        token: User authentication token.
        server: Base server URL.

    Returns:
        HTTP request response.
    """
    url, headers = build_request(
        token=token,
        server=server,
        endpoint="api/jobs/",
        headers=set([HeaderKind.AUTH, HeaderKind.ACCEPT_JSON]),
    )
    return requests.get(url=url, headers=headers)


def get_single_generator_data(token: str, generator_name: str, server: str = DEFAULT_SERVER) -> Response:
    """Get data on a given generator associated with the given token.

    Args:
        token: User authentication token.
        generator_name: Unique name of the target generator.
        server: Base server URL.

    Returns:
        HTTP request response.
    """
    url, headers = build_request(
        token=token,
        server=server,
        endpoint=f"api/jobs/{generator_name}/",
        headers=set([HeaderKind.AUTH, HeaderKind.ACCEPT_JSON]),
    )
    return requests.get(url=url, headers=headers)


def get_usage_datetime_range(
    token: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    server: str = DEFAULT_SERVER,
) -> Response:
    """Get usage stats associated with the given token over some time range.

    Args:
        token: User authentication token.
        start_time: Optional start time for query.
        end_time: Optional end time for query.
        server: Base server URL.

    Returns:
        HTTP request response.

    Raises:
        ValueError: If `end_time` is chronologically before `start_time`.
    """
    if end_time is not None and start_time is not None:
        if end_time < start_time:
            raise ValueError(f"End time ({end_time}) before start time ({start_time}) for usage query")
    query_parameters = dict()
    if start_time is not None:
        if start_time.tzinfo is None:
            start_time = start_time.astimezone()
        query_parameters["start_time"] = start_time.isoformat()
    if end_time is not None:
        if end_time.tzinfo is None:
            end_time = end_time.astimezone()
        query_parameters["end_time"] = end_time.isoformat()
    url, headers = build_request(
        token=token,
        server=server,
        endpoint="api/job_runs/counts/",
        headers=set([HeaderKind.AUTH, HeaderKind.ACCEPT_JSON]),
        query_parameters=query_parameters,
    )
    return requests.get(url=url, headers=headers)


def get_usage_last_n_days(
    token: str,
    n_days: int,
    server: str = DEFAULT_SERVER,
) -> Response:
    """Get usage stats associated with the given token for the last N days.

    Args:
        token: User authentication token.
        n_days: Number of days into the past to query usage stats for.
        server: Base server URL.

    Returns:
        HTTP request response.
    """
    end_time = datetime.now().astimezone()
    start_time = end_time - timedelta(days=n_days)
    return get_usage_datetime_range(token=token, server=server, start_time=start_time, end_time=end_time)


def post_batch(
    token: str,
    generator: str,
    name: str,
    job_params: List[Dict[str, Any]],
    is_preview: bool,
    server: str,
) -> Response:
    """Post a batch to the Infinity API.

    Args:
        token: User authentication token.
        generator: Unique name of the target generator.
        name: Descriptive name for the batch.
        job_params: List of dictionaries containing job parameters for all jobs of the batch.
        is_preview: Flag indicating the batch consists of preview job types.
        server: Base server URL.

    Returns:
        HTTP request response.

    Raises:
        ValueError: If `token` or `generator` is an empty string.
        TypeError: If `job_params` is not a `list` of `dict`s.
        ValueError: If `job_params` is an empty list.
    """
    _validate_batch_params(generator, job_params, token)
    job_runs = [{"name": generator, "is_preview": is_preview, "param_values": jp} for jp in job_params]
    json_data = {"name": name, "job_runs": job_runs}
    url, headers = build_request(
        token=token,
        server=server,
        endpoint="api/batch/",
        headers=set([HeaderKind.AUTH, HeaderKind.ACCEPT_JSON, HeaderKind.JSON_CONTENT]),
    )
    return requests.post(url=url, headers=headers, json=json_data)


def estimate_batch_samples(
    token: str,
    generator: str,
    name: str,
    job_params: List[Dict[str, Any]],
    is_preview: bool,
    server: str,
) -> Response:
    """Estimate number of samples by job in a batch.

    Args:
        token: User authentication token.
        generator: Unique name of the target generator.
        name: Descriptive name for the batch.
        job_params: List of dictionaries containing job parameters for all jobs of the batch.
        is_preview: Flag indicating the batch consists of preview job types.
        server: Base server URL.

    Returns:
        HTTP request response.

    Raises:
        ValueError: If `token` or `generator` is an empty string.
        TypeError: If `job_params` is not a `list` of `dict`s.
        ValueError: If `job_params` is an empty list.
    """
    _validate_batch_params(generator, job_params, token)
    job_runs = [{"name": generator, "is_preview": is_preview, "param_values": jp} for jp in job_params]
    json_data = {"name": name, "job_runs": job_runs}
    url, headers = build_request(
        token=token,
        server=server,
        endpoint="api/batch/estimate/",
        headers=set([HeaderKind.AUTH, HeaderKind.ACCEPT_JSON, HeaderKind.JSON_CONTENT]),
    )
    return requests.post(url=url, headers=headers, json=json_data)


def _validate_batch_params(generator: str, job_params: List[Dict[str, Any]], token: str) -> None:
    if token == "":
        raise ValueError("`token` cannot be an empty string")
    if generator == "":
        raise ValueError("`generator` cannot be an empty string")
    if not isinstance(job_params, list):
        raise TypeError(f"`job_params` must be a `list`, got {type(job_params)}")
    if len(job_params) == 0:
        raise ValueError("`job_params` is empty; no jobs to submit!")
    if not all([isinstance(d, dict) for d in job_params]):
        raise TypeError("Not all elements of `job_params` are of type `dict`")
