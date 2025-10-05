import abc
import enum
import json
import os
from pathlib import Path
from typing import *
import zipfile

import numpy as np
import pandas as pd
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from datasets import load_dataset

from glue3d.data import QADataset, load_pointcloud
from glue3d.models import AnswerGenerator


@enum.unique
class QATasks(enum.Enum):
    CAPTION = "captioning_task"
    BINARY = "binary_task"
    MULTICHOICE = "multiplechoice_task"


@enum.unique
class Datasets(enum.Enum):
    QA3D_POINTS = "GLUE3D-points"
    QA3D_POINTS_8K = "GLUE3D-points-8K"
    QA3D_IMAGE = "GLUE3D-image"
    QA3D_MULTIVIEW = "GLUE3D-multiview"
    QA3D_TEXT = "GLUE3D-text"


QUESTION_TEMPLATES = {
    QATasks.CAPTION: "{question}",
    QATasks.BINARY: "Only answer with 'Yes' or 'No': {question}",
    QATasks.MULTICHOICE: "Only answer with either A,B,C,D: {question}",
}


def download_from_repo(filename: str) -> Path:
    return hf_hub_download(
        repo_id="giorgio-mariani-1/GLUE3D",
        repo_type="dataset",
        filename=filename,
    )


def download_and_cache_zipfile(data_file: str, cache_dir: Path):
    # Download the file from the Hugging Face Hub and cache it
    path = download_from_repo(f"{data_file}.zip")

    if not (cache_dir / data_file).exists():
        with zipfile.ZipFile(path, "r") as zip_ref:
            # Check that the zip file contains the expected filenames
            for f in zip_ref.filelist:
                assert Path(f.filename).parts[0] == data_file, f"Unexpected file {f.filename} in zipfile {data_file}"

            # Extract the contents to the cache directory
            zip_ref.extractall(cache_dir)

    return cache_dir / data_file


# create pose list of 4x4 matrices, and intrisic matrix
def load_camera_parameters(camera_parameters: Dict):
    import numpy as np

    poses = []
    for frame in camera_parameters["frames"]:
        m = np.array(frame["transform_matrix"])
        m[:3, 1] = m[:3, 1] * -1  # Invert y-axis for camera coordinates
        m[:3, 2] = m[:3, 2] * -1
        poses.append(m)

    poses = np.stack(poses, axis=0)  # (V, 4, 4)
    intrinsics_kwrds = ["fl_x", "fl_y", "cx", "cy", "h", "w"]
    fl_x, fl_y, cx, cy, h, w = [camera_parameters[k] for k in intrinsics_kwrds]
    instrinsics_matrix = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]])  # (3, 3)
    return poses, instrinsics_matrix


def load_GLUE3D_benchmark(dataset_name: str, qa_task: str, cache_dir: Path) -> QADataset:
    dataset = Datasets(dataset_name)
    mode = QATasks(qa_task)

    data = load_dataset("giorgio-mariani-1/GLUE3D", mode.value, split="test")

    def load_depth_file(filepath: str) -> np.ndarray:
        import OpenEXR

        (exr_part,) = OpenEXR.File(str(filepath)).parts
        return np.array(exr_part.channels["V"].pixels).astype(np.float32).reshape(-1, 1)

    def load_multiviews(multiview_dir: Path, camera_parameters: Dict) -> Dict[str, Any]:
        depths = [load_depth_file(f) for f in sorted(multiview_dir.glob("*.exr"))]
        depths = np.stack([depths], axis=0).reshape(1, 5, 512, 512)

        data = {}
        data["images"] = [str(f) for f in sorted(multiview_dir.glob("*.png"))]
        data["depth_maps"] = depths
        data["poses"], data["intrinsics"] = load_camera_parameters(camera_parameters)
        return data

    # Setup loading function
    if dataset == Datasets.QA3D_POINTS:
        data_dir = download_and_cache_zipfile("pointclouds", cache_dir)
        load_fn = lambda x: load_pointcloud(data_dir / f"{x}.npy")
    elif dataset == Datasets.QA3D_POINTS_8K:
        data_dir = download_and_cache_zipfile("pointclouds_8192", cache_dir)
        load_fn = lambda x: load_pointcloud(data_dir / f"{x}.npy")
    elif dataset == Datasets.QA3D_IMAGE:
        data_dir = download_and_cache_zipfile("images", cache_dir)
        load_fn = lambda x: (data_dir / f"{x}.png").as_posix()
    elif dataset == Datasets.QA3D_MULTIVIEW:
        data_dir = download_and_cache_zipfile("multiviews", cache_dir)
        cam_params = json.load(open(data_dir / "transforms.json"))
        load_fn = lambda x: load_multiviews(data_dir / f"{x}", cam_params)
    elif dataset == Datasets.QA3D_TEXT:
        datafile = download_from_repo("annotations/captions_minimal.csv")
        df_captions = pd.read_csv(datafile, index_col="OBJECT_ID")
        load_fn = lambda x: df_captions.loc[x]["CAPTION"]
    else:
        assert False

    return QADataset(
        benchmark_questions=data,
        load_fn=load_fn,
        question_template=QUESTION_TEMPLATES[mode],
    )


def process_binary(answer: str):
    if answer is not ["Yes", "No"]:
        raise ValueError(f"Output answer '{answer}' should be either 'Yes' or 'No', found {answer} instead!")

    return answer == "Yes"


def process_multichoice(answer: str):
    choices = ["A", "B", "C", "D"]
    if answer not in choices:
        raise ValueError(f"Output answer '{answer}' should be one of {choices}, found {type(answer)} instead!")

    return answer


def process_caption(answer: str):
    if not isinstance(answer, str):
        raise ValueError(f"Output answer '{answer}' should be a string, found {type(answer)} instead!")
    return answer


def generate_GLUE3D_answers(
    qa_task: str,
    dataset_type: str,
    answer_generator: AnswerGenerator,
    output_file: Optional[Path] = None,
) -> pd.DataFrame:
    output_file = Path(output_file)

    # Task to answer checker
    processors = {
        QATasks.BINARY: process_binary,
        QATasks.MULTICHOICE: process_multichoice,
        QATasks.CAPTION: process_caption,
    }

    # Check if output file exists
    if output_file is not None:
        if output_file.exists():
            raise FileExistsError(
                f"Output file {output_file} already exists. Please remove it or choose a different name."
            )
        elif output_file.parent.exists() is False:
            raise FileNotFoundError(f"Output directory {output_file.parent} does not exist. Please create it first.")
        elif output_file.suffix != ".csv":
            raise ValueError(f"Output file must have a .csv extension, got {output_file.suffix}.")

    # Load benchmark data
    cache_dir = os.environ.get("GLUE3D_CACHE_DIR", None)
    if cache_dir is None:
        cache_dir = Path(".cache/glue3d").absolute()
        print(f"Warning: 'GLUE3D_CACHE_DIR' is not set. Using default cache directory ({cache_dir}).")

    dataset = load_GLUE3D_benchmark(dataset_type, qa_task, cache_dir=cache_dir)
    task_processor = processors[QATasks(qa_task)]

    responses = []
    for batch in tqdm(dataset):

        # Get IDS
        object_id = batch["object_id"]  # <- string
        question_id = batch["question_id"]  # <- string

        # Get data
        question_data = batch["data"]  # <- tensor of shape N, C(6)
        question = batch["question"]  # string

        # Compute anser
        answer = answer_generator(question_data, question)  # List of strings
        answer = task_processor(answer)

        # Append results to output file
        responses.append({"OBJECT_ID": object_id, "QUESTION_ID": question_id, "MODEL_ANSWER": answer})

    # Save the results to a CSV file
    responses_df = pd.DataFrame.from_records(responses)
    if output_file is not None:
        responses_df.to_csv(output_file, index=False)
    return responses_df
