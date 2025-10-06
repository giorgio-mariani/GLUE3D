import abc
import enum
import json
from pathlib import Path
from typing import *
from typing import Any, Dict
import zipfile

from huggingface_hub import hf_hub_download
import numpy as np
from datasets import Dataset as HFDataset
from datasets import load_dataset
import pandas as pd


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


class Dataset(abc.ABC):
    def __init__(self): ...

    def __getitem__(self, item: int) -> Dict[str, Any]: ...

    def __len__(self) -> int: ...


class QADataset(Dataset):
    def __init__(
        self,
        benchmark_questions: HFDataset,
        question_template: str = "{question}",
        load_fn: Callable = None,
    ):
        super().__init__()
        self.question_template = question_template
        self.load_fn = load_fn
        self.data = benchmark_questions

    def __getitem__(self, item: int):
        data_entry = self.data[item]
        object_id = data_entry["OBJECT_ID"]
        question_id = data_entry["QUESTION_ID"]
        question = data_entry["QUESTION"]

        # Load pointcloud
        pointcloud = self.load_fn(object_id)

        return dict(
            data=pointcloud,
            object_id=object_id,
            question_id=question_id,
            question=self.question_template.format(question=question),
        )

    def __len__(self):
        return len(self.data)


QUESTION_TEMPLATES = {
    QATasks.CAPTION: "{question}",
    QATasks.BINARY: "Only answer with 'Yes' or 'No': {question}",
    QATasks.MULTICHOICE: "Only answer with either A,B,C,D: {question}",
}


def load_pointcloud(file_path: Path, normalize: bool = True) -> np.ndarray:
    pc = np.load(file_path)
    assert pc.ndim == 2, f"Invalid point cloud shape: {pc.shape}"

    _, num_channels = pc.shape
    assert num_channels == 6, f"Invalid point cloud shape: {pc.shape}"

    if normalize:
        xyz = pc[:, :3]
        other_feature = pc[:, 3:]

        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        m = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
        xyz = xyz / m

        pc = np.concatenate((xyz, other_feature), axis=1)
    return pc


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
