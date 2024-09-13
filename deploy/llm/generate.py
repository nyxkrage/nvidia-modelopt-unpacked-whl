# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""A wrapper over the TensorRT-LLM high level API runner."""

import json
from io import BytesIO
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
from tensorrt_llm.executor import Fifo
from tensorrt_llm.hlapi import KvCacheConfig as TRT_KvCacheConfig
from tensorrt_llm.hlapi import SamplingParams
from tensorrt_llm.hlapi.llm import LLM as TRT_LLM
from tensorrt_llm.hlapi.tokenizer import TokenizerBase, TransformersTokenizer


def _put(self, obj: Any):

    # Serialize all tensors to be lists to be compatible with python multiprocess.
    if isinstance(obj, tuple):
        obj_list = list(obj)
        if isinstance(obj_list[1], tuple):
            tensors = list(obj_list[1])
            for i, t in enumerate(tensors):
                if torch.is_tensor(t):
                    # TODO: think about ways to accelerate the context logits IPC.
                    name = f"_generate_trtllm_request_id_{obj_list[0]}_tensor_{i}"
                    shm_writer = SharedMemory(name=name, create=True, size=t.nbytes + 2048)
                    torch.save(t, shm_writer._mmap)  # type: ignore[attr-defined]
                    shm_writer.close()
                    tensors[i] = name
            obj_list[1] = tuple(tensors)
        obj = tuple(obj_list)

    if self.conn is None:
        self.setup()
    self.conn.send(obj)  # type: ignore[union-attr]


def _get(self) -> Any:
    if self.conn is None:
        self.setup()
    obj = self.conn.recv()  # type: ignore[union-attr]
    if isinstance(obj, tuple):
        obj_list = list(obj)
        if isinstance(obj_list[1], tuple):
            tensors = list(obj_list[1])
            for i, t in enumerate(tensors):
                if isinstance(t, str) and t.startswith("_generate_trtllm_request_id_"):
                    shm_reader = SharedMemory(name=t, create=False)
                    tensors[i] = torch.load(BytesIO(shm_reader.buf))
                    shm_reader.close()
                    shm_reader.unlink()
            obj_list[1] = tuple(tensors)
        obj = tuple(obj_list)
    return obj


Fifo.put = _put
Fifo.get = _get


def _sanitize_temperature_and_top_p(temperature, top_p):
    assert temperature >= 0.0, "Temperature must be greater than 0.0."

    # TRT LLM acccepts temperature values only greater than 0.0
    temperature = max(temperature, 0.001)

    kwargs = {"temperature": temperature}
    if top_p is not None:
        # cpp executor only supports topP.value() > 0.f
        top_p = 1e-4 if top_p == 0 else top_p
        kwargs["top_p"] = top_p

    return kwargs


class LLM(TRT_LLM):
    """A wrapper over the ``tensorrt_llm.hlapi.llm.LLM`` for LLM profiling and validation."""

    def __init__(
        self,
        engine_dir: Union[str, Path],
        tokenizer: Optional[Union[str, Path, TokenizerBase]] = None,
        kv_cache_config: Dict[str, Union[int, float]] = {},
    ):
        """Initializes the LLM runner class.

        Args:
            engine_dir: the directory path of the TensorRT-LLM engine.
            tokenizer: the tokenizer. For example, a tokenizer from the Huggingface model.
            kv_cache_config: the kv cache config as a dict. Please refer to
                https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-best-practices.md
        """
        with open(Path(engine_dir) / "config.json", "r") as engine_config_file:
            engine_config = json.load(engine_config_file)
            build_config = engine_config["build_config"]
            world_size = (
                engine_config.get("pretrained_config", {}).get("mapping", {}).get("world_size", 1)
            )
            max_tokens_in_paged_kv_cache = (
                build_config["max_seq_len"] * build_config["max_batch_size"] // world_size
            )
            self.gather_context_logits = build_config.get("gather_context_logits", False)

        trt_kv_cache_config = TRT_KvCacheConfig()

        # If not specified, free_gpu_memory_fraction is set to the default TRT LLM value 0.9
        trt_kv_cache_config.free_gpu_memory_fraction = kv_cache_config.get(
            "free_gpu_memory_fraction", 0.9
        )

        # If not specified, max_tokens is set to the max value calculated above.
        if "max_tokens" in kv_cache_config:
            trt_kv_cache_config.max_tokens = kv_cache_config.get(
                "max_tokens", max_tokens_in_paged_kv_cache
            )

        if tokenizer is None:
            # Assume the tokenizer is stored in the engine_dir if not specified.
            tokenizer = engine_dir

        # CustomSentencePieceTokenizer will not be recognized by hlapi, wrapping it around TransformersTokenizer
        if type(tokenizer).__name__ in ["CustomSentencePieceTokenizer"]:
            tokenizer = TransformersTokenizer(tokenizer)

        super().__init__(
            model=engine_dir,
            tokenizer=tokenizer,
            kv_cache_config=trt_kv_cache_config,
        )

    @property
    def max_input_len(self):
        """Get the max input length from the LLM instance."""
        return self.args.build_config.max_input_len

    @property
    def max_beam_width(self):
        """Get the max beam width from the LLM instance."""
        return self.args.build_config.max_beam_width

    def generate_tokens(
        self,
        prompts: Union[Iterable[str], Iterable[List[int]]],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = None,
        keep_input_prompt: bool = True,
        stop_words: List[str] = None,
    ) -> Union[List[List[int]], List[List[List[int]]]]:
        """Generates the tokens based on the input prompts.

        Args:
            prompts: The input prompts. Could be a list of strings or token lists.
            max_new_tokens: The max output token length.
            temperature: The sampling temperature.
            top_p: The nucleus sampling parameter.
            keep_input_prompt: Set to include input prommpts in the outputs.
            stop_words: A list of words that the generate stops on.

        Returns:
            a list of output token lists if max_beam_width is 1 or a 3D list with shape [batch, beam, sequence_len].
        """
        assert temperature >= 0.0, "Temperature must be greater than 0.0."

        beam_width = self.max_beam_width
        kwargs = _sanitize_temperature_and_top_p(temperature, top_p)
        sampling_config = SamplingParams(
            max_new_tokens=max_new_tokens, beam_width=beam_width, stop=stop_words, **kwargs
        )

        prompt_ids = [
            self.tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
            for prompt in prompts
        ]
        outputs = self.generate(prompt_ids, sampling_params=sampling_config)

        def _process_output_token_id(output_token_id, prompt_id, with_input, keep_input_prompt):
            if with_input == keep_input_prompt:
                return output_token_id

            elif with_input:  # and not keep_input_prompt
                return output_token_id[len(prompt_id) :]

            else:  # not with_input and keep_input_prompt:
                return prompt_id + output_token_id

        with_input = False
        output_tokens = []
        for prompt_id, output in zip(prompt_ids, outputs):
            output_token_ids = [out.token_ids for out in output.outputs]

            for output_token_id in output_token_ids:
                output_tokens.append(
                    _process_output_token_id(
                        output_token_id, prompt_id, with_input, keep_input_prompt
                    )
                )

        return (
            output_tokens
            if beam_width == 1
            else [
                output_tokens[i : i + beam_width] for i in range(0, len(output_tokens), beam_width)
            ]
        )

    def generate_text(
        self,
        prompts: Union[Iterable[str], Iterable[List[int]]],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = None,
        keep_input_prompt: bool = True,
        stop_words: List[str] = None,
    ) -> Union[List[str], List[List[str]]]:
        """Generates the text based on the input prompts.

        Args:
            prompts: The input prompts. Could be a list of strings or token lists.
            max_new_tokens: The max output token length.
            temperature: The sampling temperature
            keep_input_prompt: Set to include input prommpts in the outputs.
            stop_words: A list of words the generate will stop on.

        Returns:
            a list of output text strings if max_beam_width is 1 or a 2D list with shape [batch, beam].
        """
        beam_width = self.max_beam_width
        output_tokens = self.generate_tokens(
            prompts,
            max_new_tokens,
            temperature,
            keep_input_prompt=keep_input_prompt,
            top_p=top_p,
            stop_words=stop_words,
        )
        if beam_width == 1:
            output_text = [self.tokenizer.decode(batch) for batch in output_tokens]
        else:
            output_text = [
                [self.tokenizer.decode(beam) for beam in batch] for batch in output_tokens
            ]
        return output_text

    def generate_context_logits(
        self,
        prompts: Union[Iterable[str], Iterable[List[int]]],
        temperature: float = 1.0,
        top_p: float = None,
    ) -> List[torch.tensor]:
        """Generates the context logits based on the input prompts.

        Args:
            prompts: The input prompts. Could be a list of strings or token lists.
            temperature: The sampling temperature.
            top_p: The nucleus sampling parameter.
            keep_input_prompt: Set to include input prommpts in the outputs.

        Returns:
            a tensor list of the context_logits.
        """
        assert (
            self.gather_context_logits
        ), "Please enable gather_context_logits flag when building the engine."
        assert temperature >= 0.0, "Temperature must be greater than 0.0."

        kwargs = _sanitize_temperature_and_top_p(temperature, top_p)
        kwargs["return_context_logits"] = True

        sampling_config = SamplingParams(max_new_tokens=1, beam_width=1, **kwargs)

        prompt_ids = [
            self.tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
            for prompt in prompts
        ]
        outputs = self.generate(prompt_ids, sampling_params=sampling_config)

        return [output.context_logits for output in outputs]
