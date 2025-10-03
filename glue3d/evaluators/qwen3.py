import re
from typing import *

from glue3d.evaluators.base import MetaJudge


CAPTION_PROMPT = """Evaluate a model-generated caption against a human-provided caption (ground truth).
The former is the output from a captioning system; the latter is written by a human annotator.

Consider the two captions and provide a score representing how coherent the two sentences are with each other.
Your response should consist of a single confidence score ranging from 0 to 100.

Below are several examples of question-answer pairs along with their corresponding confidence scores:

question1: How many oranges will there be if 1/3 of them are removed?
answer from model: There will be 6 left.
answer from label: As there are 9 oranges in total, there will be 6 oranges left if 1/3 of them are removed.
confidence score: 100

question2: What is this object?
answer from model: This is a bathtub
answer from label: This is a dirty bathtub.
confidence score: 80

question3: What is this object?
answer from model: This is a bottle of water
answer from label: This is a bottle of oil
confidence score: 50

question4: What does the boy have in his right hand?
answer from model: He is holding a white cup in his right hand.
answer from label: He is holding a sword in his right hand.
confidence score: 0

Next, I will give you the following inputs:
question: What is this object?
answer from model: {model_output}
answer from label: {ground_truth}

Please remember: your output should contain **only** the confidence score, no words or punctuation, just the number.
"""


class Qwen3Judge(MetaJudge):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.prompt = CAPTION_PROMPT

    def parse_generation(self, response: str):
        return int(response)

    def judge_answer(self, ground_truth: str, model_output: str) -> Mapping[str, Any]:

        response = self.chat_complete(self.prompt.format(ground_truth=ground_truth, model_output=model_output))
        score = self.parse_generation(response)

        # save the object_id, model_output, ground_truth, and scores for each result
        return {"QWEN_SCORE": score}

    def chat_complete(self, prompt_string: str):

        messages = [
            {"role": "system", "content": "You are an LLM Judge.\n/no_think"},
            {"role": "user", "content": prompt_string},
        ]

        output = self.model.create_chat_completion(messages, temperature=0.01)
        assitant_response = output["choices"][0]["message"]["content"]

        match = re.fullmatch("<think>(?P<thought>.|\n)*</think>(\s)*(?P<response>(.|\n)*)", assitant_response)
        return match.group("response")
