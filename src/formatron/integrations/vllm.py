"""
This module integrates the vllm library by providing convenience utilities.
"""
import collections.abc
import pickle
import typing

import kbnf
import torch
from transformers import AutoTokenizer
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

from formatron.config import EngineGenerationConfig
from formatron.formatter import FormatterBuilder
from formatron.integrations.utils import (
    get_bit_mask,
    get_fastest_compatible_logits_mask_fn,
    get_original_characters,
)

__all__ = [
    "create_engine_vocabulary",
    "create_formatters_logits_processor",
    "FormattersLogitsProcessor",
]

# vLLM moved tokenizer internals across releases; this annotation does not need
# a runtime import from its private module layout.
AnyTokenizer = typing.Any

_FORMATTER_BUILDER_KEY = "formatron_formatter_builder"
_FORMATTER_CONFIG_KEY = "formatron_engine_generation_config"
_VOCAB_PROCESSORS_KEY = "formatron_vocab_processors"


class _FormatterRequestLogitsProcessor:
    def __init__(
        self,
        formatter,
        eos_token_id: int | None,
        config: EngineGenerationConfig,
        mask_logits_fn,
    ):
        self._formatter = formatter
        self._eos_token_id = eos_token_id
        self._config = config
        self._mask_logits_fn = mask_logits_fn
        self._initialized = False
        self._last_generated_token_count = 0
        self._bit_mask = None

    def __call__(
        self,
        prompt_token_ids: list[int],
        generated_tokens: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        if self._bit_mask is None or self._bit_mask.shape[0] != (logits.shape[-1] + 31) // 32:
            self._bit_mask = get_bit_mask(logits)

        if not self._initialized:
            self._initialized = True
            if self._config.reset_at_beginning:
                self._formatter.reset()
            if self._config.read_prompt:
                for token in prompt_token_ids:
                    self._formatter.accept_token(token)
        elif len(generated_tokens) > self._last_generated_token_count:
            for token in generated_tokens[self._last_generated_token_count :]:
                if self._formatter.is_completed():
                    break
                self._formatter.accept_token(token)

        self._last_generated_token_count = len(generated_tokens)

        if self._formatter.is_completed():
            if generated_tokens and self._eos_token_id is not None and generated_tokens[-1] == self._eos_token_id:
                return logits
            logits[:] = float("-inf")
            if self._eos_token_id is not None:
                logits[self._eos_token_id] = 1000
            return logits

        self._formatter.compute_allowed_tokens()
        return self._mask_logits_fn(self._bit_mask, self._formatter, logits)


class FormattersLogitsProcessor(AdapterLogitsProcessor):
    """
    vLLM V1-compatible adapter that rebuilds a per-request Formatron formatter
    from `SamplingParams.extra_args`.
    """

    @classmethod
    def validate_params(cls, sampling_params):
        extra_args = sampling_params.extra_args or {}
        if not extra_args:
            return None
        builder = extra_args.get(_FORMATTER_BUILDER_KEY)
        config = extra_args.get(_FORMATTER_CONFIG_KEY)
        if builder is not None and not isinstance(builder, (bytes, bytearray)):
            raise ValueError(
                f"`{_FORMATTER_BUILDER_KEY}` must be serialized bytes or None."
            )
        if config is not None and not isinstance(config, (bytes, bytearray)):
            raise ValueError(
                f"`{_FORMATTER_CONFIG_KEY}` must be serialized bytes or None."
            )
        return None

    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool):
        super().__init__(vllm_config, device, is_pin_memory)
        model_config = vllm_config.model_config
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_config.tokenizer,
            revision=model_config.tokenizer_revision,
            trust_remote_code=model_config.trust_remote_code,
        )
        self._default_vocab = create_engine_vocabulary(self._tokenizer)
        self._mask_logits_fn = get_fastest_compatible_logits_mask_fn()

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params):
        extra_args = params.extra_args or {}
        formatter_builder = extra_args.get(_FORMATTER_BUILDER_KEY)
        if formatter_builder is None:
            return None

        formatter_builder = pickle.loads(formatter_builder)
        config_blob = extra_args.get(_FORMATTER_CONFIG_KEY)
        config = (
            EngineGenerationConfig()
            if config_blob is None
            else pickle.loads(config_blob)
        )
        vocab_processors_blob = extra_args.get(_VOCAB_PROCESSORS_KEY)
        vocab_processors = (
            None
            if vocab_processors_blob is None
            else pickle.loads(vocab_processors_blob)
        )
        vocab = (
            self._default_vocab
            if vocab_processors is None
            else create_engine_vocabulary(self._tokenizer, vocab_processors)
        )
        formatter = formatter_builder.build(vocab, lambda tokens: self._tokenizer.decode(tokens))
        return _FormatterRequestLogitsProcessor(
            formatter,
            self._tokenizer.eos_token_id,
            config,
            self._mask_logits_fn,
        )


def create_engine_vocabulary(
    tokenizer: AnyTokenizer,
    vocab_processors: typing.Optional[list[typing.Callable]] = None,
) -> kbnf.Vocabulary:
    """
    Create a vocabulary for the KBNF engine.
    Args:
        tokenizer: The tokenizer.
        vocab_processors: List of callables with signature (token_to_char: typing.Dict[bytes, bytes])->None.
            Callables can be used to "unmangle" encoded characters to original characters. If None, processors will be auto-detected.
    """
    vocab = tokenizer.get_vocab()
    new_vocab = get_original_characters(vocab, vocab_processors)
    return kbnf.Vocabulary(
        {k: kbnf.Token(v) for k, v in new_vocab.items()},
        {v: k for k, v in vocab.items()},
    )


def create_formatters_logits_processor(
    formatter_builders: typing.Sequence[FormatterBuilder | None] | FormatterBuilder,
    configs: typing.Sequence[EngineGenerationConfig] | None = None,
    vocab_processors: typing.Optional[list[typing.Callable]] = None,
) -> list[dict[str, typing.Any]]:
    """
    Create `SamplingParams.extra_args` payloads for `FormattersLogitsProcessor`.

    The returned list is aligned with the provided formatter builders; pass each
    payload to the matching request's `SamplingParams(extra_args=...)`, and
    construct the `LLM` with `logits_processors=[FormattersLogitsProcessor]`.
    """
    if not isinstance(formatter_builders, collections.abc.Sequence):
        formatter_builders = [formatter_builders]
    if configs is None:
        configs = [EngineGenerationConfig() for _ in formatter_builders]
    assert len(configs) == len(formatter_builders), (
        f"Number of formatter builders({len(formatter_builders)}) must match "
        f"number of configs({len(configs)})"
    )
    return [
        {
            _FORMATTER_BUILDER_KEY: None
            if formatter_builder is None
            else pickle.dumps(formatter_builder),
            _FORMATTER_CONFIG_KEY: pickle.dumps(config),
            _VOCAB_PROCESSORS_KEY: None
            if vocab_processors is None
            else pickle.dumps(vocab_processors),
        }
        for formatter_builder, config in zip(formatter_builders, configs)
    ]
