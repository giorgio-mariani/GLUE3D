from typing import *
import enum


@enum.unique
class MLLMs(enum.Enum):
    POINTLLM_7B = "pointllm_7B"
    POINTLLM_13B = "pointllm_13B"
    SHAPELLM_7B = "shapellm_7B"
    SHAPELLM_13B = "shapellm_13B"
    MINIGPT3D = "minigpt3d"
    QWEN_VL = "qwen_vl"
    PHI_VISION = "phi_vision"
    LLAVA = "llava"
    LLAVA_3D = "llava_3d"
    LLAMA3 = "llama_3"
    PHI_3_5_MINI = "phi_3.5_mini"
    PHI_2 = "phi_2"
    VICUNA_v1_1_7B = "vicuna_v1.1_7b"
    VICUNA_v1_1_13B = "vicuna_v1.1_13b"
    LLAMA_MESH = "llama_mesh"


def load_pointllm_model(model_name: str):
    # * print the model_name
    print(f"[INFO] Model name: {model_name}")
    import torch
    from pointllm import PointLLMLlamaForCausalLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PointLLMLlamaForCausalLM.from_pretrained(
        model_name,
        use_cache=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    return model, tokenizer


def load_minigpt3D_model():
    from pathlib import Path
    import os
    import argparse

    # Get working directory and parent directory
    pwd = Path(os.getcwd()).absolute()
    root_dir = Path(__file__).parent.parent.parent.absolute()

    # Change working directory to MiniGPT-3D
    os.chdir(root_dir / "MiniGPT-3D")

    from minigpt4.common.eval_utils import init_model  # <- required here due to local path in the module

    tmp = argparse.Namespace()
    tmp.cfg_path = "./eval_configs/benchmark_evaluation_paper.yaml"
    tmp.options = []
    model = init_model(tmp)
    model.eval()

    # Change working directory back to the original
    os.chdir(pwd)
    return [model]


def load_shapellm(model_path: str):
    import torch
    from transformers import AutoTokenizer
    from llava.model import LlavaLlamaForCausalLM
    from llava.constants import DEFAULT_POINT_PATCH_TOKEN, DEFAULT_PT_START_TOKEN, DEFAULT_PT_END_TOKEN

    # model_name = get_model_name_from_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path, use_cache=True, torch_dtype=torch.float16, device_map="auto"
    )

    # Add token_ids to tokenizer
    mm_use_pt_start_end = getattr(model.config, "mm_use_pt_start_end", False)
    mm_use_pt_patch_token = getattr(model.config, "mm_use_pt_patch_token", True)

    if mm_use_pt_patch_token:
        tokenizer.add_tokens([DEFAULT_POINT_PATCH_TOKEN], special_tokens=True)

    if mm_use_pt_start_end:
        tokenizer.add_tokens([DEFAULT_PT_START_TOKEN, DEFAULT_PT_END_TOKEN], special_tokens=True)

    model.resize_token_embeddings(len(tokenizer))
    vision_tower = model.get_vision_tower()

    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=model.device, dtype=torch.float16)

    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"

    return model, tokenizer


def load_qwenVL_model(quantize: bool = False):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    if quantize:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config = None

    model_card = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_card,
        torch_dtype="auto",
        device_map="auto",
        quantization_config=quantization_config,
    )
    print(model.dtype)

    processor = AutoProcessor.from_pretrained(model_card)
    return model, processor


def load_phi_vision_model(quantize: bool = False):
    from transformers import AutoProcessor, AutoModelForCausalLM, DynamicCache

    if quantize:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config = None

    model_id = "microsoft/Phi-3.5-vision-instruct"

    # Note: set _attn_implementation='eager' if you don't have flash_attn installed
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
        torch_dtype="auto",
        _attn_implementation="eager",
    )

    # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=4)
    return model, processor


def load_llava_model():
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    model_id = "llava-hf/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def load_llava3d_model():
    import torch
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import get_model_name_from_path

    disable_torch_init()

    model_name = get_model_name_from_path("ChaimZhu/LLaVA-3D-7B")
    tokenizer, model, processor, context_len = load_pretrained_model(
        "ChaimZhu/LLaVA-3D-7B", None, model_name, torch_dtype=torch.bfloat16
    )
    return model, tokenizer, processor


def load_llm(
    model_card: str,
    quantize: bool = False,
    dtype: Any = "bfloat16",
    device_map: Any = "auto",
    chat_template: Optional[str] = None,
):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if quantize:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config = None

    tokenizer = AutoTokenizer.from_pretrained(model_card)

    if chat_template is not None:
        tokenizer.chat_template = chat_template

    model = AutoModelForCausalLM.from_pretrained(
        model_card,
        quantization_config=quantization_config,
        torch_dtype=dtype,
        device_map=device_map,
    )
    return model, tokenizer


MODEL_LOADERS = {
    MLLMs.POINTLLM_7B: lambda: load_pointllm_model("RunsenXu/PointLLM_7B_v1.2"),
    MLLMs.POINTLLM_13B: lambda: load_pointllm_model("RunsenXu/PointLLM_13B_v1.2"),
    MLLMs.MINIGPT3D: load_minigpt3D_model,
    MLLMs.SHAPELLM_7B: lambda: load_shapellm("qizekun/ShapeLLM_7B_general_v1.0"),
    MLLMs.SHAPELLM_13B: lambda: load_shapellm("qizekun/ShapeLLM_13B_general_v1.0"),
    MLLMs.QWEN_VL: lambda: load_qwenVL_model(True),
    MLLMs.PHI_VISION: lambda: load_phi_vision_model(True),
    MLLMs.LLAVA: load_llava_model,
    MLLMs.LLAVA_3D: load_llava3d_model,
    MLLMs.LLAMA3: lambda: load_llm("meta-llama/Meta-Llama-3-8B-Instruct"),
    MLLMs.PHI_3_5_MINI: lambda: load_llm("microsoft/Phi-3.5-mini-instruct"),
    MLLMs.PHI_2: lambda: load_llm("microsoft/Phi-2", chat_template=PHI2_CHAT_TEMPLATE),
    MLLMs.VICUNA_v1_1_7B: lambda: load_llm("lmsys/vicuna-7b-v1.1", chat_template=VICUNA_v1_1_CHAT_TEMPLATE),
    MLLMs.VICUNA_v1_1_13B: lambda: load_llm("lmsys/vicuna-13b-v1.1", chat_template=VICUNA_v1_1_CHAT_TEMPLATE),
    MLLMs.LLAMA_MESH: lambda: load_llm("Zhengyi/LLaMA-Mesh"),
}


PHI2_CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ 'Instruct: ' + content.strip() + '\n' }}{% elif message['role'] == 'assistant' %}{{ 'Output: '  + content.strip() + '\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Output:' }}{% endif %}"
VICUNA_v1_1_CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{{ system_message }}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'].strip() + '\n' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'].strip() + eos_token + '\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"
