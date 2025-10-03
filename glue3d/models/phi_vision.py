from pathlib import Path
from typing import *

from glue3d.models.base import AnswerGenerator


class PhiVisionAnswerGenerator(AnswerGenerator):
    def __init__(
        self,
        model,
        processor,
        **kwargs,
    ):
        self.model = model
        self.processor = processor
        self.kwargs = kwargs

    def prepare_inputs(self, data: str, text: str) -> Any:
        from PIL import Image

        if isinstance(data, dict):
            images = [Image.open(img_file) for img_file in data["images"]]
        elif isinstance(data, (str, Path)):
            images = [Image.open(data)]
        else:
            assert False
        image_tokens = "\n".join([f"<|image_{i}|>" for i, _ in enumerate(images, 1)])
        qa_msg = {"role": "user", "content": f"{image_tokens}\n{text}"}
        prompt = self.processor.tokenizer.apply_chat_template([qa_msg], tokenize=False, add_generation_prompt=True)
        return self.processor(prompt, images, return_tensors="pt").to(self.model.device)

    def __call__(self, data: Any, text: str) -> str:
        import torch

        with torch.inference_mode():
            # Inference: Generation of the output
            inputs = self.prepare_inputs(data, text)
            output_ids = self.model.generate(
                **inputs,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                **self.kwargs,
                use_cache=False,  # otherwise it launches an error with DynamicCache
            )

            generated_ids = [
                output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            outputs = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            (output,) = outputs
            return output.strip()


class MultichoicePhiVsionAnswerGenerator(PhiVisionAnswerGenerator):
    def __init__(self, model, processor, choices, postprocess_fn=None, **kwargs):
        super().__init__(model, processor, **kwargs)
        self.post_process_fn = postprocess_fn if postprocess_fn is not None else lambda x: x

        self.choice_ids = {}
        for c in choices:
            (c_token,) = self.processor.tokenizer.encode(c, add_special_tokens=False)
            self.choice_ids[c] = c_token

    def __call__(self, data: Any, text: str) -> str:
        import torch

        with torch.inference_mode():
            # Inference: Generation of the output
            inputs = self.prepare_inputs(data, text)
            model_output = self.model(**inputs, return_dict=True, **self.kwargs)

            # Generate outputs
            answer_prob = {}
            logits = model_output.logits[:, -1:].cpu()  # <- [B, N, D]
            probs = torch.softmax(logits, dim=-1)
            for choice, choice_id in self.choice_ids.items():
                answer_prob[choice] = probs[:, :, choice_id].item()

            return self.post_process_fn(max(answer_prob, key=answer_prob.get))
