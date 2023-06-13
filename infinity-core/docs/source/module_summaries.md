# Module Summaries

## api

This module provides lightweight Python wrapping of the Infinity AI REST API endpoints. Use this module to directly interact with the REST API or to build higher level abstractions for interfacing with the REST API. For example, the `session` module provides a higher level `Session` abstraction for interacting with the Infinity API.

```{image} https://static1.smartbear.co/swagger/media/assets/images/swagger_logo.svg
:target: https://api.toinfinity.ai/api/schema/swagger-ui/
:width: 250
```

A [Swagger UI is available](https://api.toinfinity.ai/api/schema/swagger-ui/) for the Infinity API. The [OpenAPI](https://www.openapis.org/) schema can also be obtained through the `api` module:

```python
import infinity_core.api

schema = api.unwrap_text_payload(api.get_openapi_schema(token="MY_TOKEN"))
print(schema)
```

This documents our OpenAPI The Swagger UI provides a web-based interface to perform low-level interactions with the REST endpoint through the browser. A valid authentication token is required.

### Basic Usage

```python
from infinity_core import api as api

my_token = "MY_TOKEN"
r = api.get_all_generator_data(my_token)
assert r.ok

r = api.post_standard_job(token=token, json_data={"name": "visionfit", "param_values": {}})
assert r.ok
```

## session

This module provides a higher level `Session` API for interacting with the Infinity API. A `Session` instance can be created by passing in a user authentication token and a specific target generator for that session. This will create a session interface for the chosen generator, simplifying the process of submitting synthetic data requests and providing ergonomic functionality such as default and random job values for the target generator, validation of job parameters for the target generator, and easier synthetic data batch submission. Synthetic data generation with a session always involves producing `Batch` instances detailed in the `batch` module.

### Basic Usage

```python
from infinity_core.session import Session

my_token = "MY_TOKEN"
sesh = Session(token=token, generator="visionfit")

from pprint import pprint
pprint(sesh.parameter_info)

job_params = [{"camera_height": v} for v in [1.0, 1.5, 2.0]]
errors = sesh.validate_job_params(job_params=job_params)
assert all[e is None for e in errors]

videos = sesh.submit(job_params=job_params)
videos.await_completion()
videos.download(path="tmp/three_videos")
```

## batch

This module provides a `Batch` data structure and associated functionality to abstract over the concept of batch submission/generation for Infinity synthetic data. A batch (a logical grouping of specific jobs) is the base unit of synthetic data generation in the Infinity API. A single preview or generator job is simply a batch with one element. Use this module's abstractions to generate, track, and manipulate batches of synthetic data.

### Basic Usage

```python
 from infinity_core.batch import Batch, submit_batch

 def make_interesting_param_distribution(generator: str = "visionfit") -> Dict[str, Any]:
     # TODO: Construct a list of job parameters, sweeping and/or modifying parameters as desired.
     return dict()

 my_token = "MY_TOKEN"

 generator = "visionfit"
 batch = submit_batch(
     token=token,
     generator=generator,
     job_type=JobType.STANDARD,
     job_params=make_interesting_param_distribution(generator),
     name="demo batch",
)
 valid_completed_jobs = batch.await_completion()
 print(completed_jobs)
```

## data_structures

This module contains common or important data structures used in other `infinity-api` modules.
