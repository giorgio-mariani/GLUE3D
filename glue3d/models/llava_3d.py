import re
from typing import *

from glue3d.models.llava import LlaVaAnswerGenerator


class LlaVa3DAnswerGenerator(LlaVaAnswerGenerator):
    def __init__(
        self,
        model,
        tokenizer,
        processor,
        **kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor["video"]
        self.kwargs = kwargs

    def prepare_inputs(self, data: Any, text: str) -> Dict[str, Any]:
        import torch
        import numpy as np
        from PIL import Image
        from llava.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
            IMAGE_PLACEHOLDER,
        )
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_special_token

        model = self.model
        tokenizer = self.tokenizer
        processor = self.processor
        dtype = torch.bfloat16
        conv_mode = "llava_v1"
        device = model.device
        num_frames = len(data["images"])

        qs = text

        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        do_rescale = True
        do_normalize = True
        do_depth_scale = True
        depth_scale = 1000

        images, depth_images = [], []
        for image_file, depth_map in zip(data["images"], data["depth_maps"]):
            image = Image.open(image_file).convert("RGB")
            image_size = image.size
            image = processor.image_processor.preprocess(
                images=image, do_rescale=do_rescale, do_normalize=do_normalize, return_tensors="pt"
            )["pixel_values"][
                0
            ]  # [3, H, W]

            # from numpy array to PIL image (two byte precision)
            depth_map[depth_map > 2**16] = 0
            depth_map = (depth_map * depth_scale).astype(np.uint16)
            depth_map = Image.fromarray(depth_map, mode="I;16")

            # get depth map
            depth_image, resize_shape = processor.preprocess_depth_image(
                depth_map, do_depth_scale=do_depth_scale, depth_scale=depth_scale
            )
            depth_image = torch.as_tensor(np.ascontiguousarray(depth_image)).float()  # [H, W]

            # get pose
            images.append(image)
            depth_images.append(depth_image)

            # Prepare intrinsics
            intrinsic = processor.preprocess_instrinsic(data["intrinsics"], image_size, resize_shape)
            intrinsic = torch.from_numpy(intrinsic).float()
            intrinsic = intrinsic.unsqueeze(0).repeat(num_frames, 1, 1)

            # Prepare tensor
            images_tensor = torch.stack(images, dim=0).to(device, dtype=dtype)
            depths_tensor = torch.stack(depth_images, dim=0).to(device, dtype=dtype)
            poses_tensor = torch.from_numpy(data["poses"]).to(device, dtype=dtype)
            intrinsics_tensor = intrinsic.to(device, dtype=dtype)
            clicks_tensor = torch.zeros((0, 3)).to(device, dtype=dtype)

        input_ids = tokenizer_special_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0)

        return input_ids.to(device), dict(
            images=images_tensor,
            depths=depths_tensor,
            poses=poses_tensor,
            intrinsics=intrinsics_tensor,
            clicks=clicks_tensor,
            image_sizes=None,
            # do_sample=False,
            use_cache=True,
        )

    def __call__(self, data: Any, text: str) -> str:
        import torch

        with torch.inference_mode():
            # Inference: Generation of the output
            input_ids, inputs = self.prepare_inputs(data, text)

            output_ids = self.model.generate(
                input_ids,
                **inputs,
                **self.kwargs,
            )
            return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            # Decoding
            generated_ids = [output_ids[len(in_ids) :] for in_ids, output_ids in zip(input_ids, output_ids)]
            outputs = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            (output,) = outputs
        return output.strip()


class MultichoiceLlaVa3DAnswerGenerator(LlaVa3DAnswerGenerator):
    def __init__(self, model, tokenizer, processor, choices, postprocess_fn=None, **kwargs):
        super().__init__(model, tokenizer, processor, **kwargs)
        self.post_process_fn = postprocess_fn if postprocess_fn is not None else lambda x: x

        self.choice_ids = {}
        for c in choices:
            (c_token,) = self.tokenizer.encode(c, add_special_tokens=False)
            self.choice_ids[c] = c_token

    def __call__(self, data: Any, text: str) -> str:
        import torch

        with torch.inference_mode():
            # Inference: Generation of the output
            input_ids, inputs = self.prepare_inputs(data, text)
            model_output = self.model(input_ids, **inputs, return_dict=True, **self.kwargs)

            # Generate outputs
            answer_prob = {}
            logits = model_output.logits[:, -1:]  # <- [B, N, D]
            probs = torch.softmax(logits, dim=-1)
            for choice, choice_id in self.choice_ids.items():
                answer_prob[choice] = probs[:, :, choice_id].item()

            return self.post_process_fn(max(answer_prob, key=answer_prob.get))


# if __name__ == "__main__":
#    eval_model("","",32)
