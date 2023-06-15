import os
import re
import random
from typing import Optional, Dict, TypeVar, List, Any
from collections import defaultdict

from infinity_core.api import get_single_preview_job_data, get_single_standard_job_data
from infinity_core.batch import Batch
from infinity_core.session import Session

T = TypeVar("T")


def _clip(value: T, lower: T, higher: T) -> T:
    if value < lower:
        return lower
    elif value > higher:
        return higher
    else:
        return value


def sample_input(
        sesh: Session,
        scene: Optional[str] = None,
        exercise: Optional[str] = None,
        gender: Optional[str] = None,
        num_reps: Optional[int] = None,
        rel_baseline_speed: Optional[float] = None,
        max_rel_speed_change: Optional[float] = None,
        kinematic_noise_factor: Optional[float] = None,
        camera_distance: Optional[float] = None,
        camera_height: Optional[float] = None,
        avatar_identity: Optional[int] = None,
        randomize_skin_tone: Optional[bool] = None,
        relative_height: Optional[float] = None,
        relative_weight: Optional[float] = None,
        relative_camera_yaw_deg: Optional[float] = None,
        relative_camera_pitch_deg: Optional[float] = None,
        lighting_power: Optional[float] = None,
        relative_avatar_angle_deg: Optional[float] = None,
        frame_rate: Optional[int] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        state: Optional[str] = None,
        inter_rep_delay_avg: Optional[float] = None,
        inter_rep_delay_var: Optional[float] = None,
        add_pauses: Optional[bool] = None,
        add_wall_art: Optional[bool] = None,
        wall_art_type: Optional[str] = None,
        outfit: Optional[int] = None,
        hide_shoes: Optional[bool] = None,
        add_socks: Optional[bool] = None,
        allowable_occlusion: Optional[float] = None,
        enable_sensor_noise: Optional[bool] = None,
) -> Dict:
    """
    VisionFit input parameter sampler function

    Uses sane distributions for random sampling of VisionFit input parameters.
    """
    scene_options = sesh.parameter_info["scene"]["options"]["choices"]
    if scene is None:
        scene = str(random.choice(scene_options))
    else:
        if scene not in scene_options:
            raise ValueError(f"`scene` ({scene}) not in supported scene list ({scene_options})")

    def _exercise_to_category(exercise_name: str):
        """Maps user-facing exercise name to exercise category."""
        exercise_name = exercise_name.split("-")[0]
        category = re.split("_\d+", exercise_name)[0]
        return category

    exercise_options = sesh.parameter_info["exercise"]["options"]["choices"]
    exercise_category_to_seeds = defaultdict(list)
    for e in exercise_options:
        exercise_category_to_seeds[_exercise_to_category(e)].append(e)
    if exercise is None:
        exercise = str(random.choice(exercise_options))
    elif exercise in exercise_category_to_seeds.keys():
        exercise = str(random.choice(exercise_category_to_seeds[exercise]))
    else:
        if exercise not in exercise_options:
            supported_options = sorted(exercise_options + list(exercise_category_to_seeds.keys()))
            raise ValueError(f"`exercise` ({exercise}) not in supported list ({supported_options})")

    gender_options = sesh.parameter_info["gender"]["options"]["choices"]
    if gender is None:
        gender = str(random.choice(gender_options))
    else:
        if gender not in gender_options:
            raise ValueError(f"`gender` ({gender}) not in supported gender list ({gender_options})")

    nr_min = sesh.parameter_info["num_reps"]["options"]["min"]
    nr_max = sesh.parameter_info["num_reps"]["options"]["max"]
    if num_reps is None:
        num_reps = int(random.randint(1, 10))
    else:
        if not (nr_min <= num_reps <= nr_max):
            raise ValueError(f"`num_reps` ({num_reps}) must be in range [{nr_min}, {nr_max}]")

    rbs_min = sesh.parameter_info["rel_baseline_speed"]["options"]["min"]
    rbs_max = sesh.parameter_info["rel_baseline_speed"]["options"]["max"]
    if rel_baseline_speed is None:
        rel_baseline_speed = float(random.uniform(0.8, 1.25))
    else:
        if not (rbs_min <= rel_baseline_speed <= rbs_max):
            raise ValueError(f"`rel_baseline_speed` ({rel_baseline_speed}) must be in range [{rbs_min}, {rbs_max}]")

    mrsc_min = sesh.parameter_info["max_rel_speed_change"]["options"]["min"]
    mrsc_max = sesh.parameter_info["max_rel_speed_change"]["options"]["max"]
    if max_rel_speed_change is None:
        max_rel_speed_change = float(random.uniform(0.0, 0.2))
    else:
        if not (mrsc_min <= max_rel_speed_change <= mrsc_max):
            raise ValueError(
                f"`max_rel_speed_change` ({max_rel_speed_change}) must be in range [{mrsc_min}, {mrsc_max}]"
            )

    knf_min = sesh.parameter_info["kinematic_noise_factor"]["options"]["min"]
    knf_max = sesh.parameter_info["kinematic_noise_factor"]["options"]["max"]
    if kinematic_noise_factor is None:
        kinematic_noise_factor = 1.0
    else:
        if not (knf_min <= kinematic_noise_factor <= knf_max):
            raise ValueError(
                f"`kinematic_noise_factor` ({kinematic_noise_factor}) must be in range [{knf_min}, {knf_max}]"
            )

    cd_min = sesh.parameter_info["camera_distance"]["options"]["min"]
    cd_max = sesh.parameter_info["camera_distance"]["options"]["max"]
    if camera_distance is None:
        camera_distance = float(random.uniform(2.5, 5.0))
    else:
        if not (cd_min <= camera_distance <= cd_max):
            raise ValueError(f"`camera_distance` ({camera_distance}) must be in range [{cd_min}, {cd_max}]")

    ch_min = sesh.parameter_info["camera_height"]["options"]["min"]
    ch_max = sesh.parameter_info["camera_height"]["options"]["max"]
    if camera_height is None:
        camera_height = float(random.uniform(0.1, 2.75))
    else:
        if not (ch_min <= camera_height <= ch_max):
            raise ValueError(f"`camera_height` ({camera_height}) must be in range [{ch_min}, {ch_max}]")

    ai_min = sesh.parameter_info["avatar_identity"]["options"]["min"]
    ai_max = sesh.parameter_info["avatar_identity"]["options"]["max"]
    if avatar_identity is None:
        avatar_identity = int(random.randint(ai_min, ai_max))
    else:
        if not (ai_min <= avatar_identity <= ai_max) or not isinstance(avatar_identity, int):
            raise ValueError(f"`avatar_identity` ({avatar_identity}) must be integer in range [{ai_min}, {ai_max}]")

    if randomize_skin_tone is None:
        randomize_skin_tone = False
    else:
        if randomize_skin_tone not in [True, False]:
            raise ValueError(f"`randomize_skin_tone` ({randomize_skin_tone}) must be boolean `True` or `False`]")

    rh_min = sesh.parameter_info["relative_height"]["options"]["min"]
    rh_max = sesh.parameter_info["relative_height"]["options"]["max"]
    if relative_height is None:
        relative_height = float(_clip(random.gauss(0.0, 1.0), rh_min, rh_max))
    else:
        if not (rh_min <= relative_height <= rh_max):
            raise ValueError(f"`relative_height` ({relative_height}) must be in range [{rh_min}, {rh_max}]")

    rw_min = sesh.parameter_info["relative_weight"]["options"]["min"]
    rw_max = sesh.parameter_info["relative_weight"]["options"]["max"]
    if relative_weight is None:
        relative_weight = float(_clip(random.gauss(0.0, 1.0), rw_min, rw_max))
    else:
        if not (rw_min <= relative_weight <= rw_max):
            raise ValueError(f"`relative_weight` ({relative_weight}) must be in range [{rw_min}, {rw_max}]")

    rcyd_min = sesh.parameter_info["relative_camera_yaw_deg"]["options"]["min"]
    rcyd_max = sesh.parameter_info["relative_camera_yaw_deg"]["options"]["max"]
    if relative_camera_yaw_deg is None:
        relative_camera_yaw_deg = float(random.uniform(-15.0, 15.0))
    else:
        if not (rcyd_min <= relative_camera_yaw_deg <= rcyd_max):
            raise ValueError(
                f"`relative_camera_yaw_deg` ({relative_camera_yaw_deg}) must be in range [{rcyd_min}, {rcyd_max}]"
            )

    rcpd_min = sesh.parameter_info["relative_camera_pitch_deg"]["options"]["min"]
    rcpd_max = sesh.parameter_info["relative_camera_pitch_deg"]["options"]["max"]
    if relative_camera_pitch_deg is None:
        relative_camera_pitch_deg = float(random.uniform(-10.0, 10.0))
    else:
        if not (rcpd_min <= relative_camera_pitch_deg <= rcpd_max):
            raise ValueError(
                f"`relative_camera_pitch_deg` ({relative_camera_pitch_deg}) must be in range [{rcpd_min}, {rcpd_max}]"
            )

    lp_min = sesh.parameter_info["lighting_power"]["options"]["min"]
    lp_max = sesh.parameter_info["lighting_power"]["options"]["max"]
    if lighting_power is None:
        lighting_power = float(random.uniform(10.0, 1000.0))
    else:
        if not (lp_min <= lighting_power <= lp_max):
            raise ValueError(f"`lighting_power` ({lighting_power}) must be in range [{lp_min}, {lp_max}]")

    raad_min = sesh.parameter_info["relative_avatar_angle_deg"]["options"]["min"]
    raad_max = sesh.parameter_info["relative_avatar_angle_deg"]["options"]["max"]
    if relative_avatar_angle_deg is None:
        relative_avatar_angle_deg = float(random.uniform(-180.0, 180.0))
    else:
        if not (raad_min <= relative_avatar_angle_deg <= raad_max):
            raise ValueError(
                f"`relative_avatar_angle_deg` ({relative_avatar_angle_deg}) must be in range [{raad_min}, {raad_max}]"
            )

    frame_rate_options = sesh.parameter_info["frame_rate"]["options"]["choices"]
    if frame_rate is None:
        frame_rate = int(random.choice(frame_rate_options))
    else:
        if frame_rate not in frame_rate_options:
            raise ValueError(f"`frame_rate` ({frame_rate}) not in supported frame rate list ({frame_rate_options})")

    iw_min = sesh.parameter_info["image_width"]["options"]["min"]
    iw_max = sesh.parameter_info["image_width"]["options"]["max"]
    ih_min = sesh.parameter_info["image_height"]["options"]["min"]
    ih_max = sesh.parameter_info["image_height"]["options"]["max"]
    if image_width is None and image_height is None:
        square_size = int(random.choice([256, 512]))
        image_width = square_size
        image_height = square_size
    elif image_width is None:
        if not (ih_min <= image_height <= ih_max):
            raise ValueError(f"`image_height` ({image_height}) must be in range [{ih_min}, {ih_max}]")
        image_width = image_height
    elif image_height is None:
        if not (iw_min <= image_width <= iw_max):
            raise ValueError(f"`image_width` ({image_width}) must be in range [{iw_min}, {iw_max}]")
        image_height = image_width
    else:
        if not (ih_min <= image_height <= ih_max):
            raise ValueError(f"`image_height` ({image_height}) must be in range [{ih_min}, {ih_max}]")
        if not (iw_min <= image_width <= iw_max):
            raise ValueError(f"`image_width` ({image_width}) must be in range [{iw_min}, {iw_max}]")

    irda_min = sesh.parameter_info["inter_rep_delay_avg"]["options"]["min"]
    irda_max = sesh.parameter_info["inter_rep_delay_avg"]["options"]["max"]
    if inter_rep_delay_avg is None:
        inter_rep_delay_avg = float(random.uniform(0.0, 2.0))
    else:
        if not (irda_min <= inter_rep_delay_avg <= irda_max):
            raise ValueError(
                f"`inter_rep_delay_avg` ({inter_rep_delay_avg}) must be in range [{irda_min}, {irda_max}]"
            )

    irdv_min = sesh.parameter_info["inter_rep_delay_var"]["options"]["min"]
    irdv_max = sesh.parameter_info["inter_rep_delay_var"]["options"]["max"]
    if inter_rep_delay_var is None:
        inter_rep_delay_var = float(random.uniform(0.0, 2.0))
    else:
        if not (irdv_min <= inter_rep_delay_var <= irdv_max):
            raise ValueError(
                f"`inter_rep_delay_var` ({inter_rep_delay_var}) must be in range [{irdv_min}, {irdv_max}]"
            )

    if add_pauses is None:
        add_pauses = random.choice([True, False])
    else:
        if add_pauses not in [True, False]:
            raise ValueError(f"`add_pauses` ({add_pauses}) must be boolean `True` or `False`]")

    if add_wall_art is None:
        add_wall_art = random.choice([True, False])
    else:
        if add_wall_art not in [True, False]:
            raise ValueError(f"`add_wall_art` ({add_wall_art}) must be boolean `True` or `False`]")

    wall_art_type_options = sesh.parameter_info["wall_art_type"]["options"]["choices"]
    if wall_art_type is None:
        wall_art_type = str(random.choice(wall_art_type_options))
    else:
        if wall_art_type not in wall_art_type_options:
            raise ValueError(
                f"`wall_art_type` ({wall_art_type}) not in supported type list ({wall_art_type_options})"
            )

    otf_min = sesh.parameter_info["outfit"]["options"]["min"]
    otf_male_max = 4
    otf_female_max = 4
    if outfit is None:
        if gender.lower() == "male":
            outfit = int(random.randint(otf_min, otf_male_max))
        else:
            outfit = int(random.randint(otf_min, otf_female_max))
    else:
        if gender.lower() == "male":
            if not (otf_min <= outfit <= otf_male_max):
                raise ValueError(f"`outfit` ({outfit}, {gender}) must be in range [{otf_min}, {otf_male_max}]")
        else:
            if not (otf_min <= outfit <= otf_female_max):
                raise ValueError(f"`outfit` ({outfit}, {gender}) must be in range [{otf_min}, {otf_female_max}]")

    if hide_shoes is None:
        hide_shoes = random.choice([True, False])
    else:
        if hide_shoes not in [True, False]:
            raise ValueError(f"`hide_shoes` ({hide_shoes}) must be boolean `True` or `False`]")

    if add_socks is None:
        add_socks = random.choice([True, False])
    else:
        if add_socks not in [True, False]:
            raise ValueError(f"`add_socks` ({add_socks}) must be boolean `True` or `False`]")

    alo_min = sesh.parameter_info["allowable_occlusion"]["options"]["min"]
    alo_max = sesh.parameter_info["allowable_occlusion"]["options"]["max"]
    if allowable_occlusion is None:
        allowable_occlusion = float(random.uniform(0.0, 50.0))
    else:
        if not (alo_min <= allowable_occlusion <= alo_max):
            raise ValueError(
                f"`allowable_occlusion` ({allowable_occlusion}) must be in range [{alo_min}, {alo_max}]"
            )

    if enable_sensor_noise is None:
        enable_sensor_noise = True
    else:
        if enable_sensor_noise not in [True, False]:
            raise ValueError(f"`enable_sensor_noise` ({enable_sensor_noise}) must be boolean `True` or `False`]")

    params_dict = {
        "scene": scene,
        "exercise": exercise,
        "gender": gender,
        "num_reps": num_reps,
        "rel_baseline_speed": rel_baseline_speed,
        "max_rel_speed_change": max_rel_speed_change,
        "kinematic_noise_factor": kinematic_noise_factor,
        "camera_distance": camera_distance,
        "camera_height": camera_height,
        "avatar_identity": avatar_identity,
        "randomize_skin_tone": randomize_skin_tone,
        "relative_height": relative_height,
        "relative_weight": relative_weight,
        "relative_camera_yaw_deg": relative_camera_yaw_deg,
        "relative_camera_pitch_deg": relative_camera_pitch_deg,
        "lighting_power": lighting_power,
        "relative_avatar_angle_deg": relative_avatar_angle_deg,
        "frame_rate": frame_rate,
        "image_width": image_width,
        "image_height": image_height,
        "inter_rep_delay_avg": inter_rep_delay_avg,
        "inter_rep_delay_var": inter_rep_delay_var,
        "add_pauses": add_pauses,
        "add_wall_art": add_wall_art,
        "wall_art_type": wall_art_type,
        "outfit": outfit,
        "hide_shoes": hide_shoes,
        "add_socks": add_socks,
        "allowable_occlusion": allowable_occlusion,
        "enable_sensor_noise": enable_sensor_noise,
    }

    if state is not None:
        params_dict["state"] = state

    return params_dict

class ApiError(Exception):
    pass


def get_all_preview_images(folders: List[str]) -> List[str]:
    return [os.path.join(f, "video_preview.png") for f in folders]


def _get_job_params_from_id(token: str, job_id: str, is_preview: bool, server: str) -> Dict[str, Any]:
    try:
        if is_preview:
            r = get_single_preview_job_data(token=token, preview_id=job_id, server=server)
        else:
            r = get_single_standard_job_data(token=token, standard_job_id=job_id, server=server)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise ApiError(f"Failed to retrieve information for job with `job_id`: {job_id}") from e

    return data["param_values"]


def _is_preview_seed(token: str, seed_id: str, server: str) -> bool:
    r_preview = get_single_preview_job_data(token=token, preview_id=seed_id, server=server)
    if r_preview.ok:
        return True
    r_standard = get_single_standard_job_data(token=token, standard_job_id=seed_id, server=server)
    if r_standard.ok:
        return False

    raise ValueError(f"Job ID: {seed_id} is not a valid previous job ID for this user")


def _expand_overrides_across_each_base(
    sesh: Session,
    base_state_ids: List[str],
    override_params: List[Dict[str, Any]],
    is_preview: bool,
) -> List[Dict[str, Any]]:
    valid_parameter_set = set(sesh.parameter_info.keys())
    for override_dict in override_params:
        if not all([k in valid_parameter_set for k in override_dict.keys()]):
            raise ValueError("Not all override parameters are supported")

    params_with_overrides = []
    for seed in base_state_ids:
        original_params = _get_job_params_from_id(
            token=sesh.token, job_id=seed, is_preview=is_preview, server=sesh.server
        )
        params_with_overrides.extend([{**original_params, **op, **{"state": seed}} for op in override_params])

    return params_with_overrides


def expand_overrides_across_each_preview_state(
    sesh: Session,
    base_state_ids: List[str],
    override_params: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return _expand_overrides_across_each_base(
        sesh=sesh,
        base_state_ids=base_state_ids,
        override_params=override_params,
        is_preview=True,
    )


def expand_overrides_across_each_video_state(
    sesh: Session,
    base_state_ids: List[str],
    override_params: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return _expand_overrides_across_each_base(
        sesh=sesh,
        base_state_ids=base_state_ids,
        override_params=override_params,
        is_preview=False,
    )


def _submit_rerun_batch_with_overrides(
    sesh: Session,
    is_preview_rerun: bool,
    is_preview_seed: bool,
    base_state_id: str,
    override_params: List[Dict],
    batch_name: Optional[str] = None,
) -> Batch:
    valid_parameter_set = set(sesh.parameter_info.keys())
    for override_dict in override_params:
        if not all([k in valid_parameter_set for k in override_dict.keys()]):
            raise ValueError("Not all override parameters are supported")

    original_params = _get_job_params_from_id(
        token=sesh.token, job_id=base_state_id, is_preview=is_preview_seed, server=sesh.server
    )
    params_with_overrides = [{**original_params, **op} for op in override_params]
    for pdict in params_with_overrides:
        pdict["state"] = base_state_id

    return sesh.submit(
        job_params=params_with_overrides, is_preview=is_preview_rerun, batch_name=batch_name
    )


def submit_rerun_batch_with_overrides_previews(
    sesh: Session,
    base_state_id: str,
    override_params: List[Dict],
    batch_name: Optional[str] = None,
) -> Batch:
    return _submit_rerun_batch_with_overrides(
        sesh=sesh,
        is_preview_rerun=True,
        is_preview_seed=_is_preview_seed(token=sesh.token, seed_id=base_state_id, server=sesh.server),
        base_state_id=base_state_id,
        override_params=override_params,
        batch_name=batch_name,
    )


def submit_rerun_batch_with_overrides_videos(
    sesh: Session,
    base_state_id: str,
    override_params: List[Dict],
    batch_name: Optional[str] = None,
) -> Batch:
    return _submit_rerun_batch_with_overrides(
        sesh=sesh,
        is_preview_rerun=False,
        is_preview_seed=_is_preview_seed(token=sesh.token, seed_id=base_state_id, server=sesh.server),
        base_state_id=base_state_id,
        override_params=override_params,
        batch_name=batch_name,
    )
