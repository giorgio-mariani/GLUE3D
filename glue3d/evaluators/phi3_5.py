import re
from typing import *

from glue3d.evaluators.base import MetaJudge

CAPTION_PROMPT = """Evaluate a model-generated caption against a human-generated caption (ground truth) for a 3D model. Identify the aspects mentioned in the human caption and calculate the percentage of these aspects correctly mentioned or partially matched in the model caption. Score from 0 to 100, where each aspect contributes equally to the score. Consider similar concepts for partial score.

Provide your score (0-100) and a short justification (less than 15 words) in the format of 'score#reason'

Example:
Human: A white brown skeleton
Model: This is a 3D model of a small, cartoon-like robot. It has a spherical body and is covered in a layer of white dust.
Output: 50#mention white; skeleton and robot have similar appearence.

Now score the following:
Human: {ground_truth}
Model: {model_output}
Output: """


class Phi3Judge(MetaJudge):
    def __init__(self, model, tokenizer, max_new_tokens: int = 128):
        super().__init__()

        self.prompt = CAPTION_PROMPT
        self.tokenizer = tokenizer
        self.model = model
        self.max_new_tokens = max_new_tokens

    def chat_complete(self, prompt_string: str):
        import torch

        with torch.no_grad():
            self.model.eval()
            device = self.model.device

            inputs = self.tokenizer(prompt_string, return_tensors="pt")
            _, seq_len = inputs["input_ids"].shape
            output_ids = self.model.generate(**inputs.to(device), max_new_tokens=self.max_new_tokens)
            output_ids = output_ids[0, seq_len:]
            output_string = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return output_string

    def parse_generation(self, response: str):
        # * use regular expression to extract
        pattern = r"(\d*#.*)"
        match = re.search(pattern, response)

        try:
            response = match.group(1).strip()
            score, reason = response.split("#")

            score = int(score)
            if not (0 <= score <= 100):
                raise Exception()

        except Exception as e:
            print(f"Error: unale to parse {response}.")
            score = None
            reason = response

        return score, reason

    def judge_answer(self, ground_truth: str, model_output: str) -> Mapping[str, Any]:

        response = self.chat_complete(self.prompt.format(ground_truth=ground_truth, model_output=model_output))
        score, reason = self.parse_generation(response)

        return {"PHI_3_5_SCORE": score, "REASON": reason}
