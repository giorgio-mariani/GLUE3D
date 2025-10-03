# GLUE3D

Official implementation for *"GLUE3D: General Language Understanding Evaluation for 3D Point Clouds."*.

## Installation

Installation from source:
```bash
pip install git+git://github.com/giorgio-mariani/GLUE3D.git
```

## Usage
After installing the package, it is possible to programmatically run glue3d on your custom model by implementing the simple `glue3d.AnswerGenerator` interface. Example code is provided below:
```python
import numpy as np
from glue3d import AnswerGenerator, generate_GLUE3D_answers, evaluate_GLUE3D_answers

# Example class of a 3D-LLM for the Binary task
class YourAnswerGenerator(AnswerGenerator):
    def __init__(self, your_model, **kwargs):
        self.model = your_model
        ...  # stuff

    def __call__(self, data: np.ndarray, text: str) -> str:
        # NOTE: 'data' is a pointcloud of shape Nx6, with the three first dimension coordinates, and the last three RGB channels in (0,1)
        # NOTE: 'text' is the question being asked to the 3D-LLM.

        answer = self.model.generate_answer(data, text) # Example call
        return answer # This must be a string!

# Load your model
your_model= ...
answer_gen = YourAnswerGenerator(your_model)

# Compute results for the binary task
qa_answers = generate_GLUE3D_answers("binary_task", "GLUE3D-points-8K", answer_gen) # Results are returned as a pandas dataframe
eval_scores = evaluate_GLUE3D_answers("binary_task", qa_answers, evaluator="binary")

eval_results.to_csv("evaluation_scores.csv")
```
