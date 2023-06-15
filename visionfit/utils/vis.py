import glob
import json
import os
from enum import Enum
from dataclasses import dataclass
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import plotly.colors as colors
import re

from pathlib import Path
from typing import Any, List, Tuple, Dict, Optional
from zipfile import ZipFile
from IPython.display import HTML, display, clear_output
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from plotly.subplots import make_subplots
import plotly
from pycocotools.coco import COCO
from visionfit.utils.videos import parse_video_frames, stack_videos


def unzip(zipped_folder: str, output_dir: str):
    """Unzips contents of zip file to output directory."""
    with ZipFile(zipped_folder, "r") as zip_ref:
        zip_ref.extractall(output_dir)


def visualize_all_labels(video_job_folder: str) -> str:
    """Visualizes labels for a video job returned from the API.

    Args:
        video_job_folder: Path to the folder containing the video job results.

    Returns:
        Path to resulting animation video.
    """
    output_directory = os.path.join(video_job_folder, "output")
    os.makedirs(output_directory, exist_ok=True)

    video_rgb_path = os.path.join(video_job_folder, "video.mp4")
    video_json_path = os.path.join(video_job_folder, "labels.json")
    zipped_video_path = os.path.join(video_job_folder, "segmentation.zip")
    job_json_path = os.path.join(video_job_folder, "job.json")
    job_params = json.load(open(job_json_path))["params"]

    video_rgb_extracted = os.path.join(video_job_folder, "video.rgb")
    os.makedirs(video_rgb_extracted, exist_ok=True)
    unzip(zipped_video_path, video_rgb_extracted)

    imgs = parse_video_frames(video_rgb_path)
    fps = int(job_params["frame_rate"])
    rep_count = parse_rep_count_from_json(video_json_path)
    coco = COCO(video_json_path)

    bounding_box_path = create_bounding_boxes_video(
        output_path=os.path.join(output_directory, "bounding_box.mp4"), imgs=imgs, fps=fps, coco=coco
    )
    skeleton_path = create_keypoint_connections_video(
        output_path=os.path.join(output_directory, "skeleton.mp4"), imgs=imgs, fps=fps, coco=coco
    )
    cuboids_path = create_cuboids_video(
        output_path=os.path.join(output_directory, "cuboids.mp4"), imgs=imgs, fps=fps, coco=coco
    )
    _3D_path = create_3D_keypoints_video(
        output_path=os.path.join(output_directory, "3D_keypoints.mp4"),
        fps=fps,
        coco=coco,
        width_in_pixels=imgs.shape[2],
        height_in_pixels=imgs.shape[1],
    )
    segmentation_path = create_segmentation_video(
        output_path=os.path.join(output_directory, "segmentation.mp4"),
        folder=video_rgb_extracted,
        fps=fps,
        image_width=imgs.shape[2],
        image_height=imgs.shape[1],
    )

    clear_output()

    row1 = [video_rgb_path, segmentation_path, bounding_box_path]
    label_videos_row_1 = stack_videos(row1, axis=2)
    row2 = [cuboids_path, skeleton_path, _3D_path]
    label_videos_row_2 = stack_videos(row2, axis=2)
    label_grid_path = stack_videos([label_videos_row_1, label_videos_row_2], axis=1)
    ts_path = animate_time_series(
        output_path=os.path.join(output_directory, "timeseries.mp4"),
        y_axis=rep_count,
        fps=fps,
        width_in_pixels=imgs.shape[1] * 2,
        height_in_pixels=imgs.shape[1] * 2,
    )
    return stack_videos([label_grid_path, ts_path], axis=2)


def parse_rep_count_from_json(json_path: str, rep_count_col: str = "rep_count_from_start") -> List[float]:
    return [x[rep_count_col] for x in json.load(open(json_path))["images"]]


def animate_time_series(
    output_path: str,
    y_axis: List[float],
    fps: int,
    dpi: int = 150,
    width_in_pixels: int = 300,
    height_in_pixels: int = 300,
    title: str = "Rep Count",
    xlabel: str = "Time [s]",
) -> str:
    """Generates time series animation.

    Args:
        output_path: Filename to save output video.
        y_axis: The time series to animate.
        fps: Frame rate of output video.
        width_in_pixels: Width of output video.
        height_in_pixels: Height of output video.
        dpi: DPI of output video.
        title: Title of output video.
        xlabel: X-axis label of output video.

    Returns:
        Path to resulting animation video.
    """

    num_frames = len(y_axis)
    interval = 1 / fps * 1000
    time = np.arange(num_frames) / fps

    fig_height = height_in_pixels / dpi
    fig_width = width_in_pixels / dpi
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), dpi=dpi)
    ax.plot(time, y_axis)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid("on")

    ylim = ax.get_ylim()
    (line,) = ax.plot([0, 0], ylim, color="red")
    fig.tight_layout()

    def update(frame: int, time: npt.NDArray, ylim: Tuple[float, float]):
        line.set_data([time[frame], time[frame]], ylim)

    anim = FuncAnimation(
        fig=fig,
        func=update,
        frames=num_frames,
        fargs=(
            time,
            ylim,
        ),
        interval=interval,
        blit=False,
    )

    anim.save(output_path)
    plt.close()
    return output_path


def create_bounding_boxes_video(
    output_path: str, imgs: npt.NDArray, fps: int, coco: Any, add_text: bool = False
) -> str:
    """Overlays bounding box annotations onto video.

    Args:
        output_path: Path to output video.
        imgs: Frames of the video.
        fps: Frame rate of input video.
        coco: COCO data.
        add_text: If True, adds category or action label above each bounding box.

    Returns:
        Path to resulting animation video.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    image_dims = (imgs.shape[2], imgs.shape[1])
    out = cv2.VideoWriter(output_path, fourcc, fps, image_dims)

    for img, img_data in zip(imgs, coco.imgs.values()):
        canvas = img.copy()
        img_id = img_data["id"]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            if "bbox" not in ann:
                continue
            x, y, w, h = tuple(np.array(ann["bbox"]).astype(int))
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color=(255, 255, 255), thickness=2)
            if add_text:
                category_name = coco.cats[ann["category_id"]]["name"]
                text = ann.get("action", category_name)
                cv2.putText(
                    img=canvas,
                    text=text,
                    org=(x, y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

        out.write(canvas)
    out.release()
    return output_path


def get_project_root() -> str:
    """Utility method to get the project root directory."""
    return Path(__file__).parent.parent


def get_armature_connections() -> Dict:
    """Returns armature connections from local json file."""
    project_root = get_project_root()
    armature_json = os.path.join(project_root, "utils", "vis_assets", "armature_connections.json")
    return json.load(open(armature_json, "r"))


def create_keypoint_connections_video(output_path: str, imgs: npt.NDArray, fps: int, coco: Any) -> str:
    """Overlays keypoint connection annotations onto video.

    Args:
        output_path: Path to output video.
        imgs: Frames of the video.
        coco: COCO data.
        job_params: Job parameters from the API.

    Returns:
        Path to resulting animation video.
    """
    kp_connections = get_armature_connections()
    image_dims = (imgs.shape[2], imgs.shape[1])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, image_dims)

    for img, img_data in zip(imgs, coco.imgs.values()):
        canvas = img.copy()
        img_id = img_data["id"]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            if "armature_keypoints" not in ann:
                continue
            keypoints = ann["armature_keypoints"]
            for parent, child in kp_connections:
                if keypoints[parent]["z"] < 0 or keypoints[child]["z"] < 0:
                    continue
                x0 = keypoints[parent]["x"]
                y0 = keypoints[parent]["y"]
                x1 = keypoints[child]["x"]
                y1 = keypoints[child]["y"]
                cv2.line(canvas, (x0, y0), (x1, y1), color=(255, 255, 255), thickness=2)
                cv2.circle(canvas, (x0, y0), radius=4, color=(255, 255, 255), thickness=-1)

        out.write(canvas)
    out.release()
    return output_path


def create_cuboids_video(output_path: str, imgs: npt.NDArray, fps: int, coco: Any) -> str:
    """Overlays cuboid annotations onto video.

    Args:
        output_path: Path to output video
        imgs: Frames of the video
        coco: COCO data
        job_params: Job parameters from the API

    Returns:
        Path to resulting animation video.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    image_dims = (imgs.shape[2], imgs.shape[1])
    out = cv2.VideoWriter(output_path, fourcc, fps, image_dims)
    cuboid_edges = [
        [0, 1],
        [0, 4],
        [0, 3],
        [1, 2],
        [2, 3],
        [3, 7],
        [2, 6],
        [1, 5],
        [4, 5],
        [5, 6],
        [6, 7],
        [4, 7],
    ]

    for img, img_data in zip(imgs, coco.imgs.values()):
        canvas = img.copy()
        img_id = img_data["id"]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            if "cuboid_coordinates" not in ann:
                continue
            cuboid_points = ann["cuboid_coordinates"]

            if "z" in cuboid_points[0]:
                all_in_front = all(
                    [(cuboid_points[e[0]]["z"] > 0 and cuboid_points[e[1]]["z"] > 0) for e in cuboid_edges]
                )
                if not all_in_front:
                    continue

            for edge in cuboid_edges:
                start_point = [cuboid_points[edge[0]]["x"], cuboid_points[edge[0]]["y"]]
                end_point = [cuboid_points[edge[1]]["x"], cuboid_points[edge[1]]["y"]]
                color = tuple([int(255 * x) for x in coco.cats[ann["category_id"]]["color"][::-1]])
                color = (255, 255, 255)
                canvas = cv2.line(
                    canvas,
                    tuple(start_point),
                    tuple(end_point),
                    color=color,
                    thickness=2,
                )
        out.write(canvas)
    out.release()
    return output_path


def create_2D_keypoints_video(output_path: str, imgs: npt.NDArray, fps: int, coco: Any) -> str:
    """Overlays 2D keypoints onto video.

    Args:
        output_path: Path to output video
        imgs: Frames of the video
        coco: COCO data
        job_params: Job parameters from the API

    Returns:
        Path to resulting animation video.
    """

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    image_dims = (imgs.shape[2], imgs.shape[1])
    out = cv2.VideoWriter(output_path, fourcc, fps, image_dims)

    for img, img_data in zip(imgs, coco.imgs.values()):
        canvas = img.copy()
        img_id = img_data["id"]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            if "armature_keypoints" not in ann:
                continue
            keypoints = ann["armature_keypoints"]
            for keypoint_name, keypoint_info in keypoints.items():
                if keypoint_name == "root":
                    continue
                if keypoint_info["z"] < 0:
                    continue
                x, y = keypoint_info["x"], keypoint_info["y"]
                cv2.circle(canvas, (x, y), radius=3, color=(255, 255, 255), thickness=-1)
        out.write(canvas)
    out.release()
    return output_path


def create_3D_keypoints_video(
    output_path: str,
    fps: int,
    coco: Any,
    dpi: int = 150,
    width_in_pixels: int = 200,
    height_in_pixels: int = 200,
) -> str:
    """Visualizes 3D keypoint annotations as video.

    Args:
        output_path: Path to output video
        fps: desired frame rate of output video
        coco: COCO data
        dpi: resolution (dots per inch)
        width_in_pixels: Width of figure, in pixels
        height_in_pixels: Height of figure, in pixels

    Returns:
        Path to resulting animation video.
    """
    fig_height = height_in_pixels / dpi
    fig_width = width_in_pixels / dpi
    figsize = (fig_width, fig_height)
    fig = plt.figure(dpi=dpi, figsize=figsize)
    ax = fig.add_subplot(projection="3d")

    global_coords = []
    for img_data in coco.imgs.values():
        img_id = img_data["id"]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            if "armature_keypoints" not in ann:
                continue
            keypoints = ann["armature_keypoints"]
            for _, keypoint_info in keypoints.items():
                global_coords.append(
                    [
                        keypoint_info["x_global"],
                        keypoint_info["y_global"],
                        keypoint_info["z_global"],
                    ]
                )
    if all(["person_idx" not in ann for ann in coco.anns.values()]):
        num_people = 1
    else:
        num_people = max([ann["person_idx"] for ann in coco.anns.values() if "person_idx" in ann]) + 1
    global_coords = np.array(global_coords).reshape(-1, num_people * len(keypoints), 3)

    def update(num: int) -> None:
        graph._offsets3d = (
            global_coords[num, :, 0],
            global_coords[num, :, 1],
            global_coords[num, :, 2],
        )

    graph = ax.scatter(global_coords[0, :, 0], global_coords[0, :, 1], global_coords[0, :, 2])

    ax.set_box_aspect(np.ptp(global_coords, axis=(0, 1)))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid("off")

    ani = animation.FuncAnimation(fig, update, global_coords.shape[0], blit=False, interval=1000 / fps)
    ani.save(output_path)
    plt.close()
    return output_path


def create_segmentation_video(output_path: str, folder: str, fps: int, image_width: int, image_height: int) -> str:
    """Creates a video of frame-wise segmentation masks.

    Args:
        output_path: Path to output video.
        folder: Folder which contains the per-frame data.
        job_params: Job parameters from the API.

    Returns:
        Path to resulting animation video.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (image_width, image_height))
    iseg_paths = sorted(glob.glob(os.path.join(glob.escape(folder), "*.iseg.png")))
    for img_path in iseg_paths:
        out.write(cv2.imread(img_path))
    out.release()
    return output_path


def summarize_batch_results_as_dataframe(batch_folder: str) -> pd.DataFrame:
    """Compiles job parameters and post-job metadata associated with a batch.

    Args:
        batch_folder: Path to batch folder containing individual job results.

    Returns:
        Dataframe containing batch job parameters and any metadata extracted
        from the resulting job annotations.
    """

    def _convert_to_float(x):
        """Handler for when the job params have been converted to strings."""
        try:
            return x.astype(float)
        except:  # noqa: E722
            return x

    rgb_jsons = glob.glob(os.path.join(glob.escape(batch_folder), "**/labels.json"), recursive=True)

    metadata = []
    for rgb_json in rgb_jsons:
        job_id = os.path.basename(os.path.dirname(rgb_json))
        full_path = os.path.join(batch_folder, job_id)
        job_params_path = os.path.join(os.path.dirname(rgb_json), "job.json")
        job_params = json.load(open(job_params_path))["params"]
        json_data = json.load(open(rgb_json))
        num_frames = len(json_data["images"])
        anns = json_data["annotations"]
        person_cat_id = {e["name"]: e["id"] for e in json_data["categories"]}["person"]
        avg_percent_in_fov = np.mean([ann["percent_in_fov"] for ann in anns if ann["category_id"] == person_cat_id])
        avg_percent_occlusion = np.mean(
            [ann["percent_occlusion"] for ann in anns if ann["category_id"] == person_cat_id]
        )
        video_metadata = {
            "job_path": full_path,
            "job_id": job_id,
            "num_frames": num_frames,
            "avg_percent_in_fov": avg_percent_in_fov,
            "avg_percent_occlusion": avg_percent_occlusion,
        }
        metadata.append({**video_metadata, **job_params})
    df = pd.DataFrame(metadata)
    df = df.apply(_convert_to_float, axis=0)
    return df


def visualize_batch_results(batch_folder: str):
    """Generates histograms of job parameters and extracted metadata for a batch."""

    def _convert_to_float(x):
        """Handler for when the job params have been converted to strings."""
        try:
            return x.astype(float)
        except:  # noqa: E722
            return x

    df = summarize_batch_results_as_dataframe(batch_folder)
    columns_to_keep = [column for column in df.columns if column not in ["job_path", "job_id", "state"]]
    df = df[columns_to_keep]
    color = colors.qualitative.Plotly[0]
    df = df.apply(_convert_to_float, axis=0)

    num_per_row = 4
    row_num = len(df.columns) // num_per_row + 1
    fig = make_subplots(rows=row_num, cols=num_per_row, subplot_titles=df.columns)

    for i, col_name in enumerate(df.columns):
        row = i // num_per_row + 1
        col = i % num_per_row + 1
        if row * col > len(df.columns):
            break

        subfig = go.Figure(data=[go.Histogram(x=df[col_name], name=col_name, nbinsx=10, marker=dict(color=color))])
        fig.add_trace(subfig.data[0], row=row, col=col)
        fig.update_layout(
            template="plotly_white",
            title=f"Summary of Batch | {Path(batch_folder).stem}",
            height=200 * row_num,
            width=300 * num_per_row,
            bargap=0.1,
            showlegend=False,
        )

    fig.show("svg")


def display_batch_results(batch_folder: str) -> None:
    """Displays batch results as dataframe."""
    metadata = summarize_batch_results_as_dataframe(batch_folder)
    display(HTML(metadata.to_html()))


def visualize_landmarks(video_path: str, landmarks: List[npt.NDArray], output_path: str) -> str:
    """Visualizes MoveNet landmarks for a single video.

    Args:
        video_path: Path to video that will be processed.
        landmarks: Numpy array of shape (num_frames x 17 x 4) containing
            predicted MoveNeet keypoints.
        output_path: Path to save rendered video.

    Returns:
        output_path
    """

    KEYPOINT_PARENT_CHILD = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (0, 5),
        (0, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 6),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (image_width, image_height))

    frame_idx = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        for parent, child in KEYPOINT_PARENT_CHILD:
            x0 = int(image_width * landmarks[frame_idx][parent][1])
            y0 = int(image_height * landmarks[frame_idx][parent][0])
            x1 = int(image_width * landmarks[frame_idx][child][1])
            y1 = int(image_height * landmarks[frame_idx][child][0])
            cv2.line(image, (x0, y0), (x1, y1), color=(255, 255, 255), thickness=1)
            cv2.circle(image, (x0, y0), radius=2, color=(176, 129, 30), thickness=-1)
        out.write(image)
        frame_idx += 1
    out.release()

    return output_path


class InfinityPlot(Enum):
    PIE = 1
    HISTOGRAM = 2


def assign_plot_type(df: pd.DataFrame, col_name: str) -> InfinityPlot:
    """Assigns a plot type to a column based on the data it contains.
    Args:
        df: DataFrame containing the column.
        col_name: Name of the column.
    Returns:
        InfinityPlot: Plot type for the column.
    """
    if len(df[col_name].unique()) == 1:
        return InfinityPlot.PIE
    else:
        return InfinityPlot.HISTOGRAM


def create_specs(df: pd.DataFrame, num_per_row: int) -> List[List[Dict]]:
    """Produce a list of specs for a grid of Plotly subplots.
    Plotly requires a specs array of arrays to specify pie plot subplots before building.
    Args:
        columns: number of columns in the grid
        num_per_row: number of rows in the grid
    Returns:
        specs: list of lists of subplot specs
    """
    specs = []
    current_row = []
    for i, col in enumerate(df.columns):
        plot_type = assign_plot_type(df, col)
        if plot_type == InfinityPlot.PIE:
            current_row.append({"type": "pie"})
        else:
            current_row.append({})

        # Add filled rows to the specs array.
        if len(current_row) == num_per_row:
            specs.append(current_row)
            current_row = []

        # Fill final row.
        if i == len(df.columns) - 1 and len(current_row) < num_per_row:
            remaining_cols = num_per_row - len(current_row)
            for _ in range(remaining_cols):
                current_row.append({})
            specs.append(current_row)

    return specs


@dataclass(frozen=True)
class HexColor:
    v: str

    def __post_init__(self):
        if re.fullmatch(r"#[A-F0-9]{6}", self.v) is None:
            raise ValueError(f"Hex color {self.v} not in format #RRGGBB")

    def __str__(self) -> str:
        return self.v


def generate_infinity_scale(
    n: int, color1: HexColor = HexColor("#E31B88"), color2: HexColor = HexColor("#337DEC")
) -> List[str]:
    """Creates a linear scale between two Infinity colors.
    Args:
        n (int): Number of colors to generate.
        color1 (HexColor): First color e.g. HexColor(#RRGGBB).
        color2 (HexColor): Second color e.g. HexColor(#RRGGBB).
    Returns:
        list: List of colors.
    """
    START = np.array(mpl.colors.to_rgb(str(color1)))
    END = np.array(mpl.colors.to_rgb(str(color2)))
    convert_to_hex = lambda x: mpl.colors.to_hex((1 - x) * START + x * END)
    colors = [convert_to_hex(i) for i in np.linspace(0, 1, n)]
    return colors


def get_tick_values_and_labels(df: pd.DataFrame, col_name: str, max_xaxis_label_size: int = 7):
    """Gets the tick labels for a column. Plotly requires tick labels to be set after adding to subplots.
    Extracting labels in this function allows for customization of the tick labels.
    Args:
        df: DataFrame containing the column.
        col_name: Name of the column.
        max_size: Maximum size of the tick labels.
    Returns:
        list: List of tick labels.
    """

    def _trim_string(s: str, max_size: int) -> str:
        if len(s) > max_size:
            return s[:max_size] + ".."
        else:
            return s

    df = df[df[col_name].notna()]
    tickvals = list(range(len(df[col_name].unique())))
    ticktext = [_trim_string(d, max_xaxis_label_size) for d in sorted(df[col_name].unique())]

    return tickvals, ticktext


def plot_infinity_histogram(
    df: pd.DataFrame, col_name: str, stratify_var: Optional[str] = None
) -> plotly.graph_objects.Figure:
    """Plots a histogram of the column.
    Args:
        df (pandas.DataFrame): DataFrame containing the column.
        col_name (str): Name of the column.
        stratify_var (str): Name of the column to use for separating the data.
    Returns:
        plotly.graph_objects.Figure: Plotly figure.
    """
    fig = go.Figure()
    df = df[df[col_name].notna()]

    # Validate that the column can be stratified by stratify_var.
    if stratify_var is not None:
        is_stratifiable = ~df[[col_name, stratify_var]].isnull().values.any()

    # Create a stacked histogram of the column.
    if stratify_var is not None and is_stratifiable:
        unique_values = sorted(df[stratify_var].unique())
        infinity_colors = generate_infinity_scale(len(unique_values))
        for i, group in enumerate(unique_values):
            group_df = df[df[stratify_var] == group]
            histogram = go.Histogram(
                x=sorted(group_df[col_name]),
                name=str(group),
                marker=dict(color=infinity_colors[i]),
                showlegend=True if col_name == stratify_var else False,
                legendgroup="group1",
            )
            fig.add_trace(histogram)

    # Create a grouped histogram of the column.
    else:
        histogram = go.Histogram(
            x=df[col_name],
            name="All",
            marker=dict(color="#0097A7"),
            showlegend=True if col_name == stratify_var else False,
            legendgroup="group1",
        )
        fig.add_trace(histogram)

    return fig


def plot_infinity_pie(df: pd.DataFrame, col_name: str, stratify_var: str) -> plotly.graph_objects.Figure:
    """Plots a pie chart of the column.
    Args:
        df (pandas.DataFrame): DataFrame containing the column.
        col_name (str): Name of the column.
        stratify_var (str): Name of the column to use for separating the data.
    Returns:
        plotly.graph_objects.Figure: Plotly figure.
    """
    df = df[df[col_name].notna()]
    labels = df.sort_values(by=col_name)[col_name].unique()
    values = df.sort_values(by=col_name).groupby(col_name).count().values[:, 0]

    if col_name == stratify_var:
        colors = generate_infinity_scale(len(labels))
    else:
        colors = generate_infinity_scale(len(labels), HexColor("#0097A7"), HexColor("#F6F6F6"))

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors, line=dict(width=0.2, color="grey")),
                sort=False,
                showlegend=True if col_name == stratify_var else False,
                legendgroup="group1",
                insidetextorientation="horizontal",
            )
        ]
    )

    # After 5, the pie charts start to get a little crammed with labels.
    if len(labels) > 5:
        fig.update_traces(textposition="inside", textinfo="percent", textfont_size=10)
    else:
        fig.update_traces(textposition="inside", textinfo="percent+label", textfont_size=10)

    fig.update_layout(uniformtext_minsize=10, uniformtext_mode="hide")
    return fig


def visualize_job_params(
    job_params: List[Dict],
    num_per_row: int = 4,
    subplot_size: int = 300,
    renderer: str = None,
    stratify_var: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_xaxis_label_size: int = 7,
    plot_title: str = "Job Parameters",
) -> plotly.graph_objects.Figure:
    """Generates histograms of job parameter distributions.
    Args:
        job_params: Job parameters to visualize.
        num_per_row: number of columns in the grid.
        subplot_size: size of the subplots (pixels).
        renderer: Plotly renderer. If None, uses the default interactive renderer.
            Note: If you want plots to persist on Github, set renderer to "png".
        stratify_var: Name of the column to use for separating the data in histograms.
        max_xaxis_label_size: Maximum size of the x-axis labels.
        output_dir: Directory to save the plot. If None, does not save the plot.
        plot_title: Title of the plot.
    Returns:
        fig: Plotly subplots.
    """

    def _convert_to_float(x: Any) -> float:
        """Handler for when numerical job params have been converted to strings."""
        if isinstance(x, bool) or isinstance(x, float):
            return x
        else:
            try:
                return x.astype(float)
            except:  # noqa: E722
                return x

    df = pd.DataFrame(job_params)
    df = df.applymap(_convert_to_float)
    df = df.iloc[:, df.columns != "state"]

    row_num = len(df.columns) // num_per_row + 1
    specs = create_specs(df, num_per_row)
    fig = make_subplots(
        rows=row_num,
        cols=num_per_row,
        subplot_titles=df.columns,
        specs=specs,
        horizontal_spacing=0.05,
        vertical_spacing=0.075,
    )

    for i, col_name in enumerate(df.columns):
        row = i // num_per_row + 1
        col = i % num_per_row + 1

        plot_type = assign_plot_type(df, col_name)
        if plot_type == InfinityPlot.PIE:
            subfig = plot_infinity_pie(df, col_name, stratify_var)
            for data in subfig.data:
                fig.append_trace(data, row=row, col=col)

        elif plot_type == InfinityPlot.HISTOGRAM:
            subfig = plot_infinity_histogram(df, col_name, stratify_var)
            for data in subfig.data:
                fig.append_trace(data, row=row, col=col)
                if isinstance(df[col_name].iloc[0], str):
                    tickvalues, ticklabels = get_tick_values_and_labels(df, col_name, max_xaxis_label_size)
                    fig.update_xaxes(
                        tickmode="array", tickangle=45, tickvals=tickvalues, ticktext=ticklabels, row=row, col=col
                    )

        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

    fig.update_layout(
        template="plotly_white",
        title=plot_title,
        height=subplot_size * row_num,
        width=subplot_size * num_per_row + subplot_size,
        barmode="stack",
        font=dict(size=12),
        legend_title_text=stratify_var,
    )

    fig.update_annotations(font_size=16)
    if output_dir is not None:
        fig.write_image(output_dir)
    fig.show(renderer)
