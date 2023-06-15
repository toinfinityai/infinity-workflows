import os
from infinity_core.session import Session

from IPython.display import FileLink
from html import escape as html_escape
from infinity_core.batch import BatchRetrievalError


class CustomFileLink(FileLink):
    html_link_str = "<a href='%s' target='_blank'>%s</a>"

    def __init__(self, path, url_prefix="", result_html_prefix="", result_html_suffix="<br>", link_text=None):
        self.path = os.fsdecode(path)
        self.url_prefix = url_prefix
        self.result_html_prefix = result_html_prefix
        self.result_html_suffix = result_html_suffix
        self.link_text = link_text

    def to_html(self):
        return self._repr_html_()

    def _format_path(self) -> str:
        fp = "".join([self.url_prefix, html_escape(self.path)])
        return "".join(
            [
                self.result_html_prefix,
                self.html_link_str
                % (fp, html_escape(self.path if self.link_text is None else self.link_text, quote=False)),
                self.result_html_suffix,
            ]
        )


def get_workflow_templates_path() -> str:
    """Gets absolute path to workflow templates"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../workflow_templates"))


def get_workflow_records_path() -> str:
    """Gets absolute path to workflow records"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../workflow_records"))


def get_datetime_str_ftm() -> str:
    """Gets default date-time string formatter"""
    return "%Y_%m_%d_T_%H_%M_%S"


def is_batch_exist(batch_id: str, token: str, generator: str, server: str):
    """Checks is batch exists and available for user"""
    try:
        Session(token=token, generator=generator, server=server).batch_from_api(batch_id=batch_id)
        is_exist = True
    except BatchRetrievalError:
        is_exist = False
    return is_exist
