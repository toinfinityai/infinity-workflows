# Getting Started

```{image} https://img.shields.io/badge/code%20style-black-000000.svg
:target: https://github.com/psf/black
```

```{image} http://www.mypy-lang.org/static/mypy_badge.svg
:target: http://mypy-lang.org
```

```{image} https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
:target: https://pycqa.github.io/isort/
```

The **infinity_core** package provides tools to interact with the [Infinity API](https://infinity.ai/api) and generate synthetic data.

## Requirements

The **infinity_core** package requires Python 3.7 or newer.

## Installation

Install from PyPI:

```console
pip install infinity_core
```

Add to a Python project with Poetry:

```console
poetry add infinity_core
```

Install from the source located on GitHub:

```console
git clone git@github.com:toinfinityai/infinity-api-dev.git
poetry install
```

## Examples

### Using a `Session` (Basic)

```python
from pprint import pprint
from infinity_core.session import Session

# Start a session with the Infinity API.
token = "TOKEN"
sesh = Session(token=token, generator="visionfit-v0.3.1")

# Submit a request for three synthetic data videos. A `batch.Batch` object is returned upon
# submission, which provides various facilities for monitoring completion and downloading
# ready results.
job_params = [{"camera_height": v} for v in [1.0, 1.5, 2.0]]
videos = sesh.submit(job_params=job_params)

# Check and see if the batch is done yet.
num_jobs_remaining = videos.get_num_jobs_remaining()
print(f"Finished yet?: {'no' if num_jobs_remaining > 0 else 'yes'}")

# Block until generation is complete. If you want to carry on with coding and/or
# submitting many batches before awaiting, simply refrain from calling
# `await_completion` until ready, or spot check with `get_num_jobs_remaining` as above.
completed_jobs = videos.await_completion()

# At any point, you can manually ask for the currently completed jobs:
completed_jobs = videos.get_valid_completed_jobs()
pprint(completed_jobs)

# Download the results.
videos.download(path="tmp/three_camera_height_sweep_videos")
```

### Using a `Session` (Advanced)

```python
from infinity_core.session import Session

# Start a session.
token = "TOKEN"
sesh = Session(token=token, name="demo", generator="visionfit-v0.3.1")

# Create a big new batch of job parameters with specific properties.
import numpy as np

# A session provides the ability to generate a "random" job, where each parameter that
# has associated constraints (`min`, `max`, `choices`) is uniformly sampled within these bounds.
# The default job (using the default value associated with each parameter) is also available.
# Either of these can be used to help shape and construct a synthetic data batch you are preparing.
random_job = sesh.random_job()
default_job = sesh.default_job
pprint(random_job)
pprint(default_job)

job_params = []
for _ in range(100):
    # Here we explicitly set various job parameters in the batch we're constructing.
    # Some parameters are set constant for all jobs and some are randomly sampled.
    params = {
        "scene": np.random.choice(["BEDROOM_2", "BEDROOM_4"]),
        "exercise": "UPPERCUT-RIGHT",
        "gender": np.random.choice(["MALE", "FEMALE"]),
        "num_reps": 5,
        "camera_height": np.random.uniform(1.0, 2.5),
        "relative_height": truncnorm(2.0, 1.0, -4.0, 4.0), # Custom truncated Normal
        "image_width": 256,
        "image_height": 256,
    }
    # For the other parameters we're not explicitly setting, we can use the Session's random or
    # default job facilities to fill them out accordingly.
    params = sesh.randomize_unspecified_params(params) # Use a uniformly randomly sampled job to set unspecified values.
    # params = {**sesh.random_job(), **params} # The above line is equivalent to this
    # params = {**sesh.default_job, **params} # Or use the default job to plug in unspecified values.
    job_params.append(params)

# Check the validity of your batch of jobs before submission. Errors can be addressed
# before attempting to submit.
errors = sesh.validate_job_params(job_params=job_params)
assert all([e is None for e in errors])

# Analyze and update job params before submission using `pandas` DataFrames.
from pandas import DataFrame
new_df = DataFrame.from_records(job_params)

# Grab jobs from an old batch submitted last week
old_uppercut_batch = sesh.batch_from_api(batch_id="UPPERCUT_BATCH_ID")

# Update the old job params to be higher resolution
old_job_params = old_uppercut_batch.job_params
for jp in old_job_params:
    jp["image_height"] = 512
    jp["image_width"] = 512
old_df = DataFrame.from_records(old_job_params)

# Filter out some jobs based on various criteria using the familiar DF UX.
# ...

# Merge our fresh params with the updated older params to make our final new batch.
from pandas import concat
merged_df = concat([new_df, old_df])
final_job_params = merged_df.to_dict("records")

# Estimate the number of frames that will be generated for the batch.
total_frame_count = sum(sesh.estimate_samples(name="frankenstein", job_params=final_job_params, is_preview=False))
print(f"Est. frame count: {total_frame_count}")

# Submit the updated and combined new batch.
videos_batch = sesh.submit(name="frankenstein", job_params=final_job_params, is_preview=False)
videos_batch.await_completion()
videos_batch.download(path="tmp/merged_new_and_old_uppercut_batch")
```

### Using a `Session` (API Utilities)

```python
from pprint import pprint
from infinity_core.session import Session

# Start a session.
token = "TOKEN"
sesh = Session(token=token, name="demo", generator="visionfit-v0.3.1")

# Print complete parameter information for the generator.
# I.e., this will display parameter names and related metadata such as the
# default value and constraints (min, max, set).
pprint(sesh.parameter_info)

# Query usage stats for the last month. This will break down your token's
# usage stats as the number of samples rendered per unique generator.
usage_stats = sesh.get_usage_stats_last_n_days(30)
pprint(usage_stats)

# Query specific batches from the last month. This will return a list of
# the batches you have submitted over the last month. You can view, analyze,
# and use as a basis for another submission.
batches_last_month = sesh.get_batches_last_n_days(30)
for name, batch_id in batches_last_month:
    print(f"{name} ({batch_id})")

# Target the third batch for a rerun.
_name, batch_id = batches_last_month[2]
third_batch = sesh.batch_from_api(batch_id=batch_id)
job_params = third_batch.job_params
for jp in job_params:
    jp["image_width": 512]
    jp["image_height": 512]

third_batch_higher_res = sesh.submit(name="higher res", job_params=job_params)
third_batch_higher_res.await_completion()
third_batch.download(path="higher_res_batch")
```

# TODO document estimation for jobs. Visionfit 0.4.0 is not public and is the
# only generator with sample estimation, so holding off for now in this package

### Using the `batch` module directly

```python
 from infinity_core.batch import Batch, submit_batch
 from infinity_core.data_structures import JobType

 my_token = "MY_TOKEN"

 generator = "visionfit-v0.3.1"
 batch = submit_batch(
     token=token,
     generator=generator,
     job_type = JobType.STANDARD,
     job_params = [{}, {}],
     name="batch module demo with two default jobs",
)
 valid_completed_jobs = batch.await_completion()
 print(completed_jobs)
```

### Using the `api` module directly

```python
from infinity_core import api

token = "MY_TOKEN" # Your authentication token from Infinity AI.

# Get parameter information for a specific VisionFit generator.
visionfit_info = api.get_single_generator_data(token=token, generator_name="visionfit-v0.3.1")
print(visionfit_info)

# Get your usage from the last 30 days.
usage_stats = api.get_usage_last_n_days(token=token, n_days=30)
print(usage_stats)

# Get detailed information for a previously submitted batch.
r = api.get_batch_data(token=TOKEN, batch_id="unique-batch-id", server=SERVER)
assert r.ok

# Post a request for a single preview using default parameters.
r = api.post_batch(
    token=TOKEN,
    generator="visionfit-v0.3.1",
    name="preview post with defaults from api module",
    job_params=[{}, {}],
    is_preview=True,
    server=SERVER
)
assert r.ok

# Post a request for three standard video jobs using default parameters.
r = api.post_batch(
    token=TOKEN,
    generator="visionfit-v0.3.1",
    name="video post from api module",
    job_params=[{"frame_rate": 6, "num_reps": 1}, {"frame_rate": 6, "num_reps": 1}],
    is_preview=False,
    server=SERVER
)
assert r.ok
```
