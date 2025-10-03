from typing import *

import numpy as np

from glue3d.models import AnswerGenerator


class ShapeLLMAnswerGenerator(AnswerGenerator):
    def __init__(self, model, tokenizer, **kwargs):
        self.model = model
        self.tokenizer = tokenizer

        self.kwargs = kwargs

        from llava.conversation import conv_templates, SeparatorStyle

        self.conv_template = conv_templates["llava_v1"].copy()
        self.stop_string = (
            self.conv_template.sep if self.conv_template.sep_style != SeparatorStyle.TWO else self.conv_template.sep2
        )

    def prepare_text(self, question: str):
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.constants import POINT_TOKEN_INDEX, DEFAULT_POINT_TOKEN, DEFAULT_PT_START_TOKEN, DEFAULT_PT_END_TOKEN
        from llava.mm_utils import tokenizer_point_token

        conv = self.conv_template.copy()

        user_role, bot_role = conv.roles
        if self.model.config.mm_use_pt_start_end:
            question = DEFAULT_PT_START_TOKEN + DEFAULT_POINT_TOKEN + DEFAULT_PT_END_TOKEN + "\n" + question
        else:
            question = DEFAULT_POINT_TOKEN + "\n" + question

        conv.append_message(user_role, question)
        conv.append_message(bot_role, None)
        prompt = conv.get_prompt()

        return tokenizer_point_token(prompt, self.tokenizer, POINT_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)

    def prepare_pointcloud(self, pts) -> "torch.Tensor":
        from llava.mm_utils import process_pts, rotation

        # pts = pts.numpy()
        pts[:, :3] = rotation(pts[:, :3], [0, 0, -90])
        pts_tensor = process_pts(pts, self.model.config).unsqueeze(0)
        return pts_tensor

    def __call__(self, data: np.ndarray, text: str) -> str:
        import torch

        with torch.inference_mode():
            texts = [text]
            assert len(texts) == 1

            (question,) = texts
            device, dtype = self.model.device, self.model.dtype

            point_clouds = self.prepare_pointcloud(data)
            input_ids = self.prepare_text(question)
            # from transformers import TextStreamer
            # streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

            # Prepare stopping criteria
            from llava.mm_utils import KeywordsStoppingCriteria

            stopping_criteria = KeywordsStoppingCriteria([self.stop_string], self.tokenizer, input_ids)

            # Generate outputs
            self.model.eval()
            output_ids = self.model.generate(
                input_ids.to(device),
                points=point_clouds.to(device=device, dtype=dtype),
                do_sample=False,
                **self.kwargs,
                # streamer=streamer,
                stopping_criteria=[stopping_criteria],
            )

            sequence_len = input_ids.shape[1]
            # assert (input_ids.cpu() != output_ids[:, :sequence_len].cpu()).sum().item() == 0
            outputs = self.tokenizer.batch_decode(output_ids[:, sequence_len:], skip_special_tokens=True)
            (output,) = outputs
            return output.strip()


class MultiChoiceShapeLLMAnswerGenerator(ShapeLLMAnswerGenerator):
    def __init__(self, model, tokenizer, choices: List[str], postprocess_fn: Optional[Callable] = None, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.choice_ids = {}
        for c in choices:
            (c_token,) = self.tokenizer.encode(c, add_special_tokens=False)
            self.choice_ids[c] = c_token
        self.post_process_fn = postprocess_fn if not postprocess_fn is None else lambda x: x

    def __call__(self, data: np.ndarray, text: str) -> str:
        import torch

        with torch.inference_mode():
            device, dtype = self.model.device, self.model.dtype

            point_clouds = self.prepare_pointcloud(data)
            assert len(point_clouds.shape) == 3
            input_ids = self.prepare_text(text)

            # Generate outputs
            self.model.eval()
            answer_prob = {}
            model_output = self.model(
                input_ids.to(device=device),
                return_dict=True,
                points=point_clouds.to(device=device, dtype=dtype),
                **self.kwargs,
            )

            logits = model_output.logits[:, -1:]  # <- [B, N, D]
            probs = torch.softmax(logits, dim=-1)
            for choice, choice_id in self.choice_ids.items():
                answer_prob[choice] = probs[:, :, choice_id].item()

            return self.post_process_fn(max(answer_prob, key=answer_prob.get))
