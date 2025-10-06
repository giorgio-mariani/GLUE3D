from pathlib import Path
import pytest
import numpy as np
from glue3d.data import Datasets, QATasks
from glue3d.generate_answers import generate_GLUE3D_answers


def _check_data(data_type, data, text: str) -> bool:
    if data_type in {Datasets.QA3D_POINTS, Datasets.QA3D_POINTS_8K}:
        assert isinstance(data, np.ndarray), type(data)
        assert len(data.shape) == 2
        assert data.shape[-1] == 6
    elif data_type == Datasets.QA3D_IMAGE:
        assert isinstance(data, str), type(data)
        assert Path(data).exists(), data
    elif data_type == Datasets.QA3D_MULTIVIEW:
        assert isinstance(data, dict), type(data)
        assert set(data.keys()).issubset({"images", "depth_maps", "poses", "intrinsics"})
    elif data_type == Datasets.QA3D_TEXT:
        assert isinstance(data, str)
    else:
        assert False
    assert isinstance(text, str)


@pytest.mark.parametrize("task_type", [QATasks.BINARY, QATasks.MULTICHOICE, QATasks.CAPTION])
@pytest.mark.parametrize(
    "data_type",
    [Datasets.QA3D_POINTS, Datasets.QA3D_POINTS_8K, Datasets.QA3D_MULTIVIEW, Datasets.QA3D_IMAGE, Datasets.QA3D_TEXT],
)
def test_qa(task_type: QATasks, data_type: Datasets):

    def fn(data, text: str) -> bool:
        _check_data(data_type, data, text)
        if task_type == QATasks.BINARY:
            return "Yes"
        elif task_type == QATasks.MULTICHOICE:
            return "A"
        elif task_type == QATasks.CAPTION:
            return "The quick brown fox jumps over the lazy dog."

    # assert False, type(task_type)
    df = generate_GLUE3D_answers(qa_task=task_type.value, dataset_type=data_type.value, answer_generator=fn)
