import os
import datetime

import papermill as pm
import ipywidgets as widgets

from http import HTTPStatus
from typing import List, Optional
from IPython.display import display, clear_output, HTML
from infinity_core.api import get_all_generator_data
from visionfit.interfaces.common import (
    get_workflow_templates_path,
    get_workflow_records_path,
    get_datetime_str_ftm,
    CustomFileLink,
)
from visionfit.utils.notebook import suppress_useless_warnings


class WorkflowInterface:
    """Class to create notebook workflows."""

    def __init__(
        self,
        enable_generator_selection: bool = True,
        enable_server_selection: bool = False,
        default_generator: str = "visionfit-v0.4.0",
        default_server: str = "https://api.toinfinity.ai/",
    ):
        suppress_useless_warnings()
        clear_output()

        self.id = datetime.datetime.now().strftime(get_datetime_str_ftm())
        self.enable_generator_selection = enable_generator_selection
        self.enable_server_selection = enable_server_selection
        self.default_generator = default_generator

        self.output_after_generators = widgets.Output()
        self.output_after_create = widgets.Output()

        self.button_generators = widgets.Button(description="Get Generators")
        self.button_generators.style.font_weight = "bolder"
        self.button_generators.on_click(self.on_button_generators)

        self.button_create = widgets.Button(description="Create Notebook")
        self.button_create.style.font_weight = "bolder"
        self.button_create.on_click(self.on_button_create)

        self.token_input = widgets.Text(
            value="",
            placeholder="Enter Infinity Token Here",
            description="Infinity Token:",
            disabled=False,
        )

        self.server_dropdown = widgets.Select(
            options=["https://api.toinfinity.ai/", "https://apidev.toinfinity.ai/", "https://apirnd.toinfinity.ai/"],
            value=default_server,
            description="Server:",
            rows=3,
            disabled=False,
        )

        self.workflow_dropdown = widgets.Select(
            options=["Submit Batch", "Download Batch"],
            value="Submit Batch",
            description="Workflow:",
            rows=3,
            disabled=False,
        )

        self.generator_dropdown = widgets.Select(
            options=[None], value=None, description="Generator:", rows=5, disabled=False
        )

        display(self.workflow_dropdown, self.token_input)
        if self.enable_server_selection:
            display(self.server_dropdown)
        if self.enable_generator_selection:
            display(self.button_generators, self.output_after_generators)
        else:
            display(self.button_create, self.output_after_create, self.output_after_generators)

    def update_id(self):
        self.id = datetime.datetime.now().strftime(get_datetime_str_ftm())

    def on_button_generators(self, _):
        self.output_after_generators.clear_output()
        self.output_after_create.clear_output()

        generator_names = self._get_allowable_generators()
        if generator_names is not None:
            generator_names.sort()
            self.generator_dropdown.options = generator_names
            self.generator_dropdown.value = generator_names[0]

            with self.output_after_generators:
                display(self.generator_dropdown, self.button_create, self.output_after_create)

    def on_button_create(self, _):
        if not self.enable_generator_selection:
            self.output_after_generators.clear_output()
        self.output_after_create.clear_output()

        if not self.enable_generator_selection:
            generator_names = self._get_allowable_generators()
            if generator_names is None:
                return
            if len(generator_names) == 0:
                with self.output_after_generators:
                    self.output_after_generators.append_stderr(
                        "Token is correct, but you do not have access to generator yet"
                    )
                    return

        self.update_id()
        workflow_path = os.path.join(get_workflow_records_path(), f"workflow_{self.id}")
        os.makedirs(workflow_path, exist_ok=True)

        params = {
            "TOKEN": self.token_input.value,
            "GENERATOR": self.generator_dropdown.value if self.enable_generator_selection else self.default_generator,
            "SERVER": self.server_dropdown.value,
        }

        # Select workflow
        time_str = datetime.datetime.now().strftime(get_datetime_str_ftm())
        if self.workflow_dropdown.value == "Submit Batch":
            input_notebook_path = os.path.join(get_workflow_templates_path(), "submit_batch_template.ipynb")
            output_notebook_path = os.path.join(workflow_path, f"submit_batch_{time_str}.ipynb")
        elif self.workflow_dropdown.value == "Download Batch":
            input_notebook_path = os.path.join(get_workflow_templates_path(), "download_batch_template.ipynb")
            output_notebook_path = os.path.join(workflow_path, f"download_batch_{time_str}.ipynb")
            params["BATCH_ID"] = ""
        else:
            self.output_after_create.append_stderr("Workflow selection not supported. Please select a workflow.")
            return

        _ = pm.execute_notebook(
            input_notebook_path,
            output_notebook_path,
            prepare_only=True,
            parameters=params,
        )

        with self.output_after_create:
            link_html = CustomFileLink(os.path.relpath(output_notebook_path, os.getcwd()), link_text="Link").to_html()
            display(
                HTML(
                    f'<div style="font-family: monospace">'
                    f"{self.workflow_dropdown.value} workflow: "
                    f"{link_html}"
                    f"</div>"
                ),

            )
            print(f"Workflow path: {os.path.dirname(os.path.abspath(output_notebook_path))}")

    def _get_allowable_generators(self) -> Optional[List[str]]:
        # TODO: Avoid usage of raw response data parsing.
        #  We might should return List[Job] from "get_all_generator_data". Is not critical thing.
        try:
            response = get_all_generator_data(token=self.token_input.value, server=self.server_dropdown.value)
        except Exception as e:
            self.output_after_generators.append_stderr(str(e))
            return None

        if response.status_code != HTTPStatus.OK:
            self.output_after_generators.append_stderr(response.text)
            return None

        return [generator["name"] for generator in response.json()]
