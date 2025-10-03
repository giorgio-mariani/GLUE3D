from typing import *

from glue3d.models.base import AnswerGenerator


class LlaVaAnswerGenerator(AnswerGenerator):
    def __init__(
        self,
        model,
        processor,
        **kwargs,
    ):
        self.model = model
        self.processor = processor
        self.kwargs = kwargs

    def prepare_inputs(self, image_path: str, text: str):
        qa_msg = {
            "role": "user",
            "content": [{"type": "image", "image": image_path}, {"type": "text", "text": text}],
        }

        # Preparation for inference
        return self.processor.apply_chat_template(
            [qa_msg], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)

    def __call__(
        self,
        data: Any,
        text: str,
    ) -> str:
        import torch

        with torch.inference_mode():
            # Inference: Generation of the output
            inputs = self.prepare_inputs(data, text)
            output_ids = self.model.generate(**inputs, **self.kwargs)

            # Decoding
            generated_ids = [
                output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            outputs = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            (output,) = outputs
        return output.strip()


class MultichoiceLlaVaAnswerGenerator(LlaVaAnswerGenerator):
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
            logits = model_output.logits[:, -1:]  # <- [B, N, D]
            probs = torch.softmax(logits, dim=-1)
            for choice, choice_id in self.choice_ids.items():
                answer_prob[choice] = probs[:, :, choice_id].item()

            return self.post_process_fn(max(answer_prob, key=answer_prob.get))
