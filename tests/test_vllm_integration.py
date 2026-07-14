import gc
import pytest

torch = pytest.importorskip("torch")
vllm = pytest.importorskip("vllm")

if not torch.cuda.is_available():
    pytest.skip("CUDA is unavailable", allow_module_level=True)

LLM = vllm.LLM
SamplingParams = vllm.SamplingParams

from formatron.formatter import FormatterBuilder
from formatron.integrations.vllm import (
    FormattersLogitsProcessor,
    create_formatters_logits_processor,
)


def test_vllm_integration(snapshot):
    prompts = [
        "Hello, my name is",
        "The future of AI is",
    ]
    f = FormatterBuilder()
    f.append_line("definitely vllm!")
    f2 = FormatterBuilder()
    f2.append_line("强大的【VLLM】！！！")
    sampling_extra_args = create_formatters_logits_processor([f, f2])
    llm = LLM(
        model="openai-community/gpt2",
        logits_processors=[FormattersLogitsProcessor],
        gpu_memory_utilization=0.8,
    )
    sampling_params = [
        SamplingParams(
            max_tokens=50,
            temperature=0.8,
            skip_special_tokens=False,
            top_p=0.95,
            extra_args=extra_args,
        )
        for extra_args in sampling_extra_args
    ]
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params, )
    # Print the outputs.
    snapshot_names = ["hello_my_name_is", "future_of_ai"]
    for snapshot_name, output in zip(snapshot_names, outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        assert f"Prompt: {prompt!r}, Generated text: {generated_text!r}" == snapshot(name=snapshot_name)
    # Clean up GPU memory
    del llm
    torch.cuda.empty_cache()
    gc.collect()


def test_vllm_integration_sparse(snapshot):
    prompts = [
        "The first prompt is",
        "The second prompt is",
        "The third prompt is",
        "The fourth prompt is",
    ]
    f1 = FormatterBuilder()
    f1.append_line("formatted with vllm!")
    f3 = FormatterBuilder()
    f3.append_line("also formatted but is slightly longer!")

    # Create a sparse array of formatter builders
    sparse_formatters = [f1, None, f3, None]

    sampling_extra_args = create_formatters_logits_processor(sparse_formatters)
    llm = LLM(
        model="openai-community/gpt2",
        logits_processors=[FormattersLogitsProcessor],
        gpu_memory_utilization=0.8,
    )
    sampling_params = [
        SamplingParams(
            max_tokens=50,
            temperature=0,
            skip_special_tokens=False,
            top_p=0.1,
            extra_args=extra_args,
        )
        for extra_args in sampling_extra_args
    ]

    outputs = llm.generate(prompts, sampling_params)
    snapshot_names = [
        "first_prompt",
        "second_prompt",
        "third_prompt",
        "fourth_prompt",
    ]
    for snapshot_name, output in zip(snapshot_names, outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        assert f"Prompt: {prompt!r}, Generated text: {generated_text!r}" == snapshot(name=snapshot_name)
    del llm
    torch.cuda.empty_cache()
    gc.collect()
