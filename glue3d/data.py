import abc
from pathlib import Path
from typing import *

import numpy as np
from datasets import Dataset as HFDataset


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
