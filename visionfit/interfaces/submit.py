import os
import datetime
import papermill as pm
import ipywidgets as widgets

from typing import List
from IPython.display import display, clear_output, HTML
from visionfit.interfaces.common import get_workflow_templates_path, get_datetime_str_ftm, CustomFileLink
from visionfit.utils.notebook import suppress_useless_warnings
from infinity_tools.visionfit.api import VisionFitSession
from infinity_core.session import ParameterValidationError
from infinity_core.batch import BatchEstimationError, BatchSubmissionError
from colorama import Fore, Style

class SubmitInterface:
    """Class to generate/interact with batch submission."""

    def __init__(self, sesh: VisionFitSession, job_params: List):
        suppress_useless_warnings()
        clear_output()

        self.batch = None
        self.sesh = sesh
        self.job_params = job_params

        self.button_submit = widgets.Button(description="Submit Batch")
        self.button_submit.on_click(self.on_button_submit)
        self.button_submit.style.font_weight = "bolder"

        self.button_confirm = widgets.Button(description="Confirm Submission")
        self.button_confirm.on_click(self.on_button_confirm)
        self.button_confirm.style.font_weight = "bolder"
        self.button_confirm.style.button_color = "#FF8A8A"

        self.type_of_jobs_dropdown = widgets.Select(
            options=["Previews", "Videos"], value="Previews", description="Type of Jobs:", rows=3, disabled=False
        )

        self.name_input = widgets.Text(
            value="", placeholder="Your descriptive batch name...", description="Batch Name:", disabled=False
        )

        display(self.type_of_jobs_dropdown, self.name_input, self.button_submit)

    def on_button_submit(self, _):
        clear_output()
        is_preview = self.type_of_jobs_dropdown.value == "Previews"
        print('Validating parameters and computing frame estimate...', flush=True)
        try:
            total_est_frames = sum(self.sesh.estimate_samples(self.job_params, is_preview=is_preview))
        except (ValueError, ParameterValidationError, BatchEstimationError) as e:
            print("\n" + Fore.RED + str(e))
            print(Style.RESET_ALL)
            return
        clear_output()
        print("Please confirm that you want to submit this batch.\n")
        print(f"Estimated number of frames to render: {total_est_frames}")
        print(f"Number of Jobs: {len(self.job_params)}\n")
        print(f"Type of Jobs: {self.type_of_jobs_dropdown.value}")
        self.button_confirm.layout.display = "block"
        display(self.button_confirm)

    def on_button_confirm(self, _):
        self.button_confirm.layout.display = "none"

        # Set type of jobs
        try:
            is_preview = self.type_of_jobs_dropdown.value == "Previews"
            self.batch = self.sesh.submit(
                job_params=self.job_params,
                is_preview=is_preview,
                batch_name=self.name_input.value,
            )
        except (ValueError, ParameterValidationError, BatchSubmissionError) as e:
            print("\n" + Fore.RED + str(e))
            print(Style.RESET_ALL)
            return
        print("\n------- Submission Successful -------")

        # Print batch information (including GUI)
        print(f"\nBatch ID: {self.batch.batch_id}")
        admin_gui_link = self.batch.server + "admin/api/batch/" + self.batch.batch_id
        display(
            HTML(
                f'<div style="font-family: monospace">'
                f'User Portal: <a href="{admin_gui_link}" target="_blank">Link</a>'
                f"</div>"
            ),
        )

        # Generate 'Download Batch' workflow
        time_str = datetime.datetime.now().strftime(get_datetime_str_ftm())
        input_notebook_path = os.path.join(get_workflow_templates_path(), "download_batch_template.ipynb")
        output_notebook_path = f"download_batch_{time_str}.ipynb"
        _ = pm.execute_notebook(
            input_notebook_path,
            output_notebook_path,
            prepare_only=True,
            parameters={
                "TOKEN": self.batch.token,
                "GENERATOR": self.sesh.generator,
                "SERVER": self.batch.server,
                "BATCH_ID": self.batch.batch_id,
            },
        )

        link_html = CustomFileLink(os.path.relpath(output_notebook_path, os.getcwd()), link_text="Link").to_html()
        display(
            HTML(
                f'<div style="font-family: monospace">' f"Download Batch workflow: {link_html}" f"</div>"
            ),
        )
        print(f"Workflow path: {os.path.dirname(os.path.abspath(output_notebook_path))}")
