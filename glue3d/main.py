from pathlib import Path
from typing import *

import click

from glue3d.generate_answers import generate_GLUE3D_answers
from glue3d.evaluate_answers import evaluate_GLUE3D_answers
from glue3d.models.loaders import MLLMs, MODEL_LOADERS
from glue3d.generate_answers import QATasks, Datasets
from glue3d.evaluators import Evaluators


def prepare_answer_generator(model: MLLMs):
    task_to_cls = {}

    if model in {MLLMs.POINTLLM_7B, MLLMs.POINTLLM_13B}:
        from glue3d.models.pointllm import PointLLMAnswerGenerator, MultiChoicePointLLMAnswerGenerator

        task_to_cls[QATasks.CAPTION] = PointLLMAnswerGenerator
        task_to_cls[QATasks.BINARY] = MultiChoicePointLLMAnswerGenerator
        task_to_cls[QATasks.MULTICHOICE] = MultiChoicePointLLMAnswerGenerator
    if model in {MLLMs.SHAPELLM_7B, MLLMs.SHAPELLM_13B}:
        from glue3d.models.shapellm import ShapeLLMAnswerGenerator, MultiChoiceShapeLLMAnswerGenerator

        task_to_cls[QATasks.CAPTION] = ShapeLLMAnswerGenerator
        task_to_cls[QATasks.BINARY] = MultiChoiceShapeLLMAnswerGenerator
        task_to_cls[QATasks.MULTICHOICE] = MultiChoiceShapeLLMAnswerGenerator
    if model == MLLMs.MINIGPT3D:
        from glue3d.models.minigpt3d import MiniGPT3DAnswerGenerator, MultiChoiceMiniGPT3DAnswerGenerator

        task_to_cls[QATasks.CAPTION] = MiniGPT3DAnswerGenerator
        task_to_cls[QATasks.BINARY] = MultiChoiceMiniGPT3DAnswerGenerator
        task_to_cls[QATasks.MULTICHOICE] = MultiChoiceMiniGPT3DAnswerGenerator
    if model == MLLMs.QWEN_VL:
        from glue3d.models.qwen import QwenAnswerGenerator, MultichoiceQwenAnswerGenerator

        task_to_cls[QATasks.CAPTION] = QwenAnswerGenerator
        task_to_cls[QATasks.BINARY] = MultichoiceQwenAnswerGenerator
        task_to_cls[QATasks.MULTICHOICE] = MultichoiceQwenAnswerGenerator
    if model == MLLMs.LLAVA:
        from glue3d.models.llava import LlaVaAnswerGenerator, MultichoiceLlaVaAnswerGenerator

        task_to_cls[QATasks.CAPTION] = LlaVaAnswerGenerator
        task_to_cls[QATasks.BINARY] = MultichoiceLlaVaAnswerGenerator
        task_to_cls[QATasks.MULTICHOICE] = MultichoiceLlaVaAnswerGenerator
    if model == MLLMs.PHI_VISION:
        from glue3d.models.phi_vision import PhiVisionAnswerGenerator, MultichoicePhiVsionAnswerGenerator

        task_to_cls[QATasks.CAPTION] = PhiVisionAnswerGenerator
        task_to_cls[QATasks.BINARY] = MultichoicePhiVsionAnswerGenerator
        task_to_cls[QATasks.MULTICHOICE] = MultichoicePhiVsionAnswerGenerator

    if model == MLLMs.LLAVA_3D:
        from glue3d.models.llava_3d import LlaVa3DAnswerGenerator, MultichoiceLlaVa3DAnswerGenerator

        task_to_cls[QATasks.CAPTION] = LlaVa3DAnswerGenerator
        task_to_cls[QATasks.BINARY] = MultichoiceLlaVa3DAnswerGenerator
        task_to_cls[QATasks.MULTICHOICE] = MultichoiceLlaVa3DAnswerGenerator

    if model in {
        MLLMs.LLAMA3,
        MLLMs.PHI_3_5_MINI,
        MLLMs.PHI_2,
        MLLMs.VICUNA_v1_1_7B,
        MLLMs.VICUNA_v1_1_13B,
        MLLMs.LLAMA_MESH,
    }:
        from glue3d.models.text_only import TextAnswerGenerator, MultichoiceTextAnswerGenerator

        task_to_cls[QATasks.CAPTION] = TextAnswerGenerator
        task_to_cls[QATasks.BINARY] = MultichoiceTextAnswerGenerator
        task_to_cls[QATasks.MULTICHOICE] = MultichoiceTextAnswerGenerator

    def binary_postprocess_fn(x):
        assert x in ["Yes", "No"]
        return x == "Yes"

    output = {}
    output[QATasks.CAPTION] = lambda *args: task_to_cls[QATasks.CAPTION](*args, max_new_tokens=128)
    output[QATasks.BINARY] = lambda *args: task_to_cls[QATasks.BINARY](
        *args, choices=["Yes", "No"], postprocess_fn=binary_postprocess_fn
    )
    output[QATasks.MULTICHOICE] = lambda *args: task_to_cls[QATasks.MULTICHOICE](*args, choices=["A", "B", "C", "D"])
    return output


@click.command("generate")
@click.option("--model", "-m", type=click.Choice([x.value for x in MLLMs]), required=True)
@click.option(
    "--task", "-t", default=[QATasks.BINARY.value], type=click.Choice([x.value for x in QATasks]), multiple=True
)
@click.option("--data", "-d", type=click.Choice([x.value for x in Datasets]), required=True)
@click.option("--output-file", "-o", default="tmp.csv", type=str)
def generate_answers(model: str, task: List[str], data: str, output_file: str = "tmp.csv"):

    model_args = MODEL_LOADERS[MLLMs(model)]()
    answer_generator_constructor = prepare_answer_generator(MLLMs(model))
    output_file = Path(output_file)

    for t in task:
        if len(task) > 1:
            outfile = output_file.with_name(f"{output_file.stem}-{t}{output_file.suffix}")
        else:
            outfile = output_file

        generate_GLUE3D_answers(
            answer_generator=answer_generator_constructor[QATasks(t)](*model_args),
            qa_task=t,
            dataset_type=data,
            output_file=outfile,
        )


@click.command("evaluate")
@click.argument("input-file", type=str)
@click.argument("output-file", type=str)
@click.option("--task", "-t", type=click.Choice([x.value for x in QATasks]), required=True)
@click.option("--evaluator", "-e", type=click.Choice([x.value for x in Evaluators]))
def evaluate_answers(input_file: str, output_file: str, task: str, evaluator: Optional[str] = None):

    if QATasks(task) == QATasks.CAPTION:
        judge = Evaluators(evaluator)
        from glue3d.evaluators.loaders import load_phi3_5_mini_instruct, load_qwen3_30B_A3B_model

        if judge == Evaluators.PHI_3_5_JUDGE:
            from glue3d.evaluators.phi3_5 import Phi3Judge

            evaluator = Phi3Judge(*load_phi3_5_mini_instruct())

        elif judge == Evaluators.QWEN_3_30B_JUDGE:
            from glue3d.evaluators.qwen3 import Qwen3Judge

            evaluator = Qwen3Judge(*load_qwen3_30B_A3B_model())
        elif judge == Evaluators.TRADITIONAL:
            from glue3d.evaluators.misc import TraditionalCaptionMetricEvaluator

            evaluator = TraditionalCaptionMetricEvaluator()
        else:
            assert False

    else:
        print("Ignoring evaluator, using exact match for evaluation.")
        evaluator = lambda **kwargs: kwargs["ANSWER"] == kwargs["MODEL_ANSWER"]

    import pandas as pd

    answers = pd.read_csv(input_file, index_col=["OBJECT_ID", "QUESTION_ID"])

    out = evaluate_GLUE3D_answers(
        model_answer_data=answers,
        ground_truth_data=task,
        answer_evaluator=evaluator,
    )

    out.to_csv(output_file)


@click.group()
def cli():
    pass


cli.add_command(generate_answers, name="generate")
cli.add_command(evaluate_answers, name="evaluate")

if __name__ == "__main__":
    cli()
