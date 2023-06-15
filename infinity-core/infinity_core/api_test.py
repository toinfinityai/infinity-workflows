import pytest

import infinity_core.api as api
from infinity_core.data_structures import HeaderKind


class TestBuildRequest:
    def test_reject_empty_token_string(self) -> None:
        token = ""
        with pytest.raises(ValueError, match="`token` cannot be an empty string"):
            api.build_request(token=token, server=api.DEFAULT_SERVER, endpoint="ep")

    def test_reject_empty_server(self) -> None:
        server = ""
        with pytest.raises(ValueError, match="`server` cannot be an empty string"):
            api.build_request(token="123", server=server, endpoint="ep")

    def test_reject_empty_endpoint(self) -> None:
        endpoint = ""
        with pytest.raises(ValueError, match="`endpoint` cannot be an empty string"):
            api.build_request(token="123", server=api.DEFAULT_SERVER, endpoint=endpoint)

    @pytest.mark.parametrize("endpoint", ["api/ep", "/api/ep", "/api/ep/", "api/ep/"])
    def test_join_server_and_endpoint(self, endpoint: str) -> None:
        token = "123"
        server = "https://api.company.com"
        url, _ = api.build_request(token=token, server=server, endpoint=endpoint)

        assert url == server + "/api/ep/"

    def test_join_query_parameters(self) -> None:
        token = "123"
        server = api.DEFAULT_SERVER
        endpoint = "ep"
        query_parameters = {"var1": "1", "var2": "2"}
        url, _ = api.build_request(token=token, server=server, endpoint=endpoint, query_parameters=query_parameters)

        assert url == server + "/ep/" + "?" + "var1=1&var2=2"

    def test_combined_headers(self) -> None:
        token = "123"
        server = api.DEFAULT_SERVER
        endpoint = "ep"
        headers_set = set([HeaderKind.AUTH, HeaderKind.ACCEPT_JSON, HeaderKind.JSON_CONTENT])
        _, headers = api.build_request(token=token, server=server, endpoint=endpoint, headers=headers_set)

        exp_headers = {
            **HeaderKind.AUTH.to_header_dict(token),
            **HeaderKind.ACCEPT_JSON.to_header_dict(token),
            **HeaderKind.JSON_CONTENT.to_header_dict(token),
        }

        assert headers == exp_headers


class TestPostBatch:
    def test_reject_empty_token_string(self) -> None:
        token = ""
        with pytest.raises(ValueError, match="`token` cannot be an empty string"):
            api.post_batch(
                token=token,
                generator="test_generator",
                name="test batch",
                job_params=[{}],
                is_preview=True,
                server=api.DEFAULT_SERVER,
            )

    def test_reject_empty_server(self) -> None:
        server = ""
        with pytest.raises(ValueError, match="`server` cannot be an empty string"):
            api.post_batch(
                token="test_token",
                generator="test_generator",
                name="test batch",
                job_params=[{}],
                is_preview=True,
                server=server,
            )

    def test_reject_empty_generator(self) -> None:
        generator = ""
        with pytest.raises(ValueError, match="`generator` cannot be an empty string"):
            api.post_batch(
                token="test_token",
                generator=generator,
                name="test batch",
                job_params=[{}],
                is_preview=True,
                server=api.DEFAULT_SERVER,
            )

    def test_reject_non_list_job_params(self) -> None:
        job_params = ({},)  #  type:ignore
        with pytest.raises(TypeError, match="`job_params` must be a `list`"):
            api.post_batch(
                token="test_token",
                generator="test_generator",
                name="test batch",
                job_params=job_params,  # type:ignore
                is_preview=True,
                server=api.DEFAULT_SERVER,
            )

    def test_reject_empty_job_params(self) -> None:
        job_params = []  # type: ignore
        with pytest.raises(ValueError, match="`job_params` is empty"):
            api.post_batch(
                token="test_token",
                generator="test_generator",
                name="test batch",
                job_params=job_params,
                is_preview=True,
                server=api.DEFAULT_SERVER,
            )

    def test_reject_list_with_non_dict_elements_in_job_params(self) -> None:
        job_params = [{}, []]  # type: ignore
        with pytest.raises(TypeError, match="Not all elements of `job_params` are of type `dict`"):
            api.post_batch(
                token="test_token",
                generator="test_generator",
                name="test batch",
                job_params=job_params,  # type: ignore
                is_preview=True,
                server=api.DEFAULT_SERVER,
            )
