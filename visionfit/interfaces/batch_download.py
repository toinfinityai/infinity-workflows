import os
import sys
import datetime
import ipywidgets as widgets

from string import Formatter
from IPython.display import display, HTML
from infinity_core.batch import DownloadError
from visionfit.interfaces.common import CustomFileLink


class BatchDownloadInterface:
    """Class to interact with batch download and visualization."""

    def __init__(self, batch, batch_path):
        self.batch = batch
        self.batch_path = batch_path

        self.button_download = widgets.Button(description="Download")
        self.button_download.style.font_weight = "bolder"
        self.button_download.on_click(self._on_button_download)

        self.output = widgets.Output()

        print(f"Batch Id: {self.batch.batch_id}\n")
        print(f"Num of Jobs: {len(self.batch.jobs)}")
        admin_gui_link = self.batch.server + "admin/api/batch/" + self.batch.batch_id
        display(
            HTML(
                f'<div style="font-family: monospace">'
                f'User Portal: <a href="{admin_gui_link}" target="_blank">Link</a>'
                f"</div>"
            ),
        )
        display(self.button_download, self.output)
        self.status_display = display(display_id=True)

    def _on_button_download(self, _):
        self.output.clear_output()

        successful_jobs = self.batch.get_valid_completed_jobs()
        completed_jobs = self.batch.get_completed_jobs()

        link_html = CustomFileLink(
            os.path.relpath(self.batch_path, os.getcwd()),
            link_text="link",
            result_html_suffix="",
        ).to_html()

        try:
            download_status = self.batch.download(path=self.batch_path, quiet=True)
        except DownloadError as e:
            download_status = False
            print(str(e))

        # Check if job still ongoing
        if len(completed_jobs) == len(self.batch.jobs):
            # Visualize the completed job.
            with self.output:
                if download_status:
                    num_failed = len(self.batch.jobs) - len(successful_jobs)
                    if num_failed > 0:
                        print(f"\033[91m {num_failed} jobs have failed")
                    display(
                        HTML(
                            f'<div style="font-family: monospace">'
                            f"Your batch of data has been downloaded.<br>"
                            f"Path to Data on Local Machine: {os.path.abspath(self.batch_path)} "
                            f"({link_html})"
                            f"</div>"
                        ),
                    )

        else:
            # Display duration since job was registered by API endpoint
            job_duration = self._calculate_job_duration()
            with self.output:
                print(f"Your data is still being generated.")
                print(f"Time elapsed: {self._duration_to_str(job_duration)}")
                print(f"{len(completed_jobs)}/{len(self.batch.jobs)} submitted jobs have completed.")

                if download_status:
                    display(
                        HTML(
                            f'<div style="font-family: monospace">'
                            f"The completed jobs in your batch have been downloaded.<br>"
                            f"Path to Data on Local Machine: {os.path.abspath(self.batch_path)} "
                            f"({link_html})"
                            f"</div>"
                        ),
                    )

    def _calculate_job_duration(self) -> datetime.timedelta:
        """Returns the duration of time since the job was registered by the API."""
        current_time = datetime.datetime.now().astimezone()
        batch_time = datetime.datetime.fromisoformat(self.batch.get_batch_data()["created"])
        delta_time = current_time - batch_time
        return delta_time

    @staticmethod
    def _duration_to_str(duration: datetime.timedelta) -> str:
        f = Formatter()
        d = {}
        ftm_pars = {"D": "{D} d ", "H": "{H} h ", "M": "{M} min ", "S": "{S} sec"}
        mods = {"D": 86400, "H": 3600, "M": 60, "S": 1}
        rem = int(duration.total_seconds())
        ftm = ""
        for i in ftm_pars.keys():
            qt, rem = divmod(rem, mods[i])
            if qt > 0:
                d[i] = qt
                ftm += ftm_pars[i]
        return f.format(ftm, **d)
