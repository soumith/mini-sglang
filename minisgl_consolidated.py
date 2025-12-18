"""
================================================================================
MINISGL - CONSOLIDATED CODEBASE (Execution-Order Edition)
================================================================================

This file presents the entire minisgl codebase (~5000 lines) reordered to follow
the actual execution flow when serving a Qwen3 model. Each section includes
annotations explaining WHEN and WHY the code is executed.

================================================================================
EXECUTION FLOW OVERVIEW
================================================================================

PROCESS ARCHITECTURE (3 processes connected via ZMQ):
─────────────────────────────────────────────────────

  ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
  │   API SERVER     │      │    TOKENIZER     │      │    SCHEDULER     │
  │   (FastAPI)      │◄────►│    (Worker)      │◄────►│    (GPU)         │
  │                  │ ZMQ  │                  │ ZMQ  │                  │
  │ - HTTP endpoints │      │ - Tokenization   │      │ - Model forward  │
  │ - Async streaming│      │ - Detokenization │      │ - KV cache mgmt  │
  └──────────────────┘      └──────────────────┘      └──────────────────┘

COMPLETE REQUEST FLOW (what happens when you call /v1/chat/completions):
────────────────────────────────────────────────────────────────────────

  CLIENT                API SERVER              TOKENIZER               SCHEDULER
    │                      │                       │                        │
    │  POST /v1/chat/      │                       │                        │
    │  completions         │                       │                        │
    │─────────────────────►│                       │                        │
    │                      │                       │                        │
    │                      │  1. Create uid        │                        │
    │                      │  2. TokenizeMsg       │                        │
    │                      │      (uid, text,      │                        │
    │                      │       params)         │                        │
    │                      │──────────────────────►│                        │
    │                      │                       │                        │
    │                      │                       │  3. tokenizer.encode() │
    │                      │                       │  4. UserMsg            │
    │                      │                       │      (uid, input_ids,  │
    │                      │                       │       params)          │
    │                      │                       │───────────────────────►│
    │                      │                       │                        │
    │                      │                       │      ┌─────────────────┤
    │                      │                       │      │ 5. PREFILL      │
    │                      │                       │      │  - match_prefix │
    │                      │                       │      │  - allocate KV  │
    │                      │                       │      │  - forward()    │
    │                      │                       │      │  - sample()     │
    │                      │                       │      └─────────────────┤
    │                      │                       │                        │
    │                      │                       │  6. DetokenizeMsg      │
    │                      │                       │      (uid, token_id,   │
    │                      │                       │       finished=False)  │
    │                      │                       │◄───────────────────────│
    │                      │                       │                        │
    │                      │  7. tokenizer.decode()│                        │
    │                      │  8. UserReply         │                        │
    │                      │      (uid, text,      │                        │
    │                      │       finished=False) │                        │
    │                      │◄──────────────────────│                        │
    │                      │                       │                        │
    │  9. SSE chunk        │                       │      ┌─────────────────┤
    │     "data: Hello"    │                       │      │ 10. DECODE LOOP │
    │◄─────────────────────│                       │      │  - forward()    │
    │                      │                       │      │  - sample()     │
    │                      │                       │      │  - repeat 6-9   │
    │  ... more chunks ... │                       │      │    until done   │
    │                      │                       │      └─────────────────┤
    │                      │                       │                        │
    │                      │                       │  11. DetokenizeMsg     │
    │                      │                       │      (finished=True)   │
    │                      │                       │◄───────────────────────│
    │                      │                       │                        │
    │                      │  12. UserReply        │                        │
    │                      │      (finished=True)  │                        │
    │                      │◄──────────────────────│                        │
    │                      │                       │                        │
    │  13. "data: [DONE]"  │                       │                        │
    │◄─────────────────────│                       │                        │
    │                      │                       │                        │

SERVER STARTUP SEQUENCE:
────────────────────────

    launch_server()
         │
         ├──► parse_args() ──► ServerArgs
         │
         ├──► spawn tokenize_worker() process
         │         │
         │         └──► AutoTokenizer.from_pretrained()
         │         └──► TokenizeManager, DetokenizeManager
         │         └──► ZMQ queues (recv from API, send to scheduler)
         │
         ├──► spawn scheduler process
         │         │
         │         └──► Scheduler.__init__()
         │               │
         │               ├──► Engine.__init__()
         │               │     ├── set_tp_info() (distributed)
         │               │     ├── create_model() (Qwen3ForCausalLM)
         │               │     ├── load_hf_weight() (from HuggingFace)
         │               │     ├── create_kvcache() (MHAKVCache)
         │               │     ├── create_attention_backend() (FA3/FI)
         │               │     └── GraphRunner.capture() (CUDA graphs)
         │               │
         │               ├──► CacheManager() (with radix tree)
         │               ├──► PrefillManager()
         │               ├──► DecodeManager()
         │               └──► SchedulerIOMixin (ZMQ setup)
         │
         └──► run_api_server()
               │
               └──► uvicorn.run(FastAPI app)

SCHEDULER MAIN LOOP (run_forever):
──────────────────────────────────

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    MAIN SERVING LOOP (per batch)                         │
    │                                                                          │
    │  1. RECEIVE MESSAGES (from tokenizer via ZMQ)                           │
    │     receive_msg(blocking=not runnable)                                  │
    │       └── For each UserMsg: prefill_manager.add_one_req(msg)            │
    │                                                                          │
    │  2. SCHEDULE NEXT BATCH                                                  │
    │     _schedule_next_batch()                                              │
    │       ├── prefill_manager.schedule_next_batch() OR                      │
    │       │   decode_manager.schedule_next_batch()                          │
    │       │                                                                  │
    │       └── _prepare_batch(batch)                                         │
    │             ├── cache_manager.match_prefix() (radix tree lookup)        │
    │             ├── cache_manager.allocate() (get KV cache pages)           │
    │             └── attn_backend.prepare_metadata()                         │
    │                                                                          │
    │  3. FORWARD PASS (GPU execution)                                         │
    │     _forward(forward_input)                                             │
    │       └── engine.forward_batch(batch, sample_args)                      │
    │             └── model.forward_batch(batch)                              │
    │                   │                                                      │
    │                   │  Qwen3ForCausalLM.forward():                         │
    │                   │    ├── VocabParallelEmbedding (token -> hidden)     │
    │                   │    ├── For each decoder layer:                      │
    │                   │    │     ├── RMSNormFused (pre-attention)           │
    │                   │    │     ├── RopeAttn.forward()                     │
    │                   │    │     │     ├── LinearQKVMerged -> Q,K,V         │
    │                   │    │     │     ├── QK-norm (Qwen3 only)             │
    │                   │    │     │     ├── AttentionLayer                   │
    │                   │    │     │     │     ├── RoPE                       │
    │                   │    │     │     │     ├── FlashInfer/FA3 attn        │
    │                   │    │     │     │     └── store_kv() -> cache        │
    │                   │    │     │     └── LinearOProj + all_reduce         │
    │                   │    │     ├── RMSNormFused (post-attention)          │
    │                   │    │     └── GatedMLP                               │
    │                   │    │           ├── gate_up_proj                     │
    │                   │    │           ├── silu_and_mul                     │
    │                   │    │           └── down_proj + all_reduce           │
    │                   │    └── ParallelLMHead -> logits                     │
    │                   │                                                      │
    │             └── sampler.sample(logits) -> next_tokens                   │
    │                                                                          │
    │  4. PROCESS RESULTS                                                      │
    │     _process_last_data(last_data, ongoing_data)                         │
    │       ├── For each request:                                             │
    │       │     ├── req.append_host(next_token)                             │
    │       │     ├── Check finish: EOS? max_tokens? max_seq_len?             │
    │       │     └── Create DetokenizeMsg(uid, token, finished)              │
    │       │                                                                  │
    │       ├── send_result(reply) -> ZMQ -> Tokenizer -> API                 │
    │       │                                                                  │
    │       └── For finished requests:                                        │
    │             ├── table_manager.free(table_idx)                           │
    │             └── cache_manager.free_and_cache_finished_req()             │
    │                   └── Insert prefix into radix tree for future reuse    │
    │                                                                          │
    │  5. REPEAT (overlap mode: GPU compute overlaps with CPU processing)     │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
TABLE OF CONTENTS (by execution order)
================================================================================

PART 1: STARTUP & CONFIGURATION
  - Entry Point (__main__.py, shell.py)
  - Server Launch (server/launch.py)
  - Argument Parsing (server/args.py)
  - Configuration Classes (engine/config.py, scheduler/config.py)
  - Environment Variables (env.py)

PART 2: API SERVER (FastAPI endpoints)
  - FrontendManager (manages async request/response)
  - HTTP endpoints (/generate, /v1/chat/completions)
  - SSE streaming responses

PART 3: TOKENIZER WORKER
  - tokenize_worker() (separate process)
  - TokenizeManager (text -> token IDs)
  - DetokenizeManager (token IDs -> text)

PART 4: CORE DATA STRUCTURES
  - SamplingParams, Req, Batch, Context (core.py)
  - Message Protocol (message/*.py)

PART 5: DISTRIBUTED COMMUNICATION
  - DistributedInfo, set_tp_info (distributed/info.py)
  - DistributedCommunicator, PyNCCL (distributed/impl.py)

PART 6: SCHEDULER (The Brain)
  - Scheduler main class (scheduler/scheduler.py)
  - I/O handling (scheduler/io.py)
  - PrefillManager (scheduler/prefill.py)
  - DecodeManager (scheduler/decode.py)
  - CacheManager (scheduler/cache.py)
  - TableManager (scheduler/table.py)

PART 7: ENGINE (The Executor)
  - Engine class (engine/engine.py)
  - CUDA Graph Runner (engine/graph.py)
  - Sampler (engine/sample.py)

PART 8: MODELS
  - ModelConfig (models/config.py)
  - BaseLLMModel (models/base.py)
  - Qwen3ForCausalLM (models/qwen3.py)
  - LlamaForCausalLM (models/llama.py)
  - Model utilities: GatedMLP, RopeAttn (models/utils.py)
  - Weight loading (models/weight.py)

PART 9: LAYERS (Building Blocks)
  - BaseOP, StateLessOP, OPList (layers/base.py)
  - Linear layers with TP (layers/linear.py)
  - Embeddings (layers/embedding.py)
  - AttentionLayer (layers/attention.py)
  - RoPE (layers/rotary.py)
  - Normalization (layers/norm.py)
  - Activation (layers/activation.py)

PART 10: KV CACHE
  - BaseKVCache, MHAKVCache (kvcache/base.py, mha_pool.py)
  - RadixCacheManager (kvcache/radix_manager.py)

PART 11: ATTENTION BACKENDS
  - BaseAttnBackend (attention/base.py)
  - FlashInfer backend (attention/fi.py)
  - FlashAttention3 backend (attention/fa3.py)

PART 12: UTILITIES
  - Logging (utils/logger.py)
  - ZMQ messaging (utils/mp.py)
  - Misc helpers (utils/misc.py)

================================================================================
"""

from __future__ import annotations
import os
import sys
import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache, partial
from typing import (
    TYPE_CHECKING, Any, Callable, Dict, Generic, List,
    Literal, NamedTuple, NoReturn, Set, Tuple, TypeAlias, TypeVar
)


# ##############################################################################
# PART 1: STARTUP & CONFIGURATION
# ##############################################################################
#
# WHEN: At process start, before any inference
# WHY: Set up the server, parse arguments, configure the system
#
# Execution order:
#   1. __main__.py calls launch_server()
#   2. launch.py parses args and spawns processes
#   3. api_server.py starts FastAPI
#   4. Scheduler/Tokenizer processes start

# ==============================================================================
# Entry Points
# ==============================================================================
# File: minisgl/__main__.py
#
# This is invoked when running `python -m minisgl`
#
# from .server import launch_server
# launch_server()
#
# File: minisgl/shell.py (for interactive mode)
#
# from .server import launch_server
# launch_server(run_shell=True)

# ==============================================================================
# Environment Variables (env.py)
# ==============================================================================
# WHEN: At import time, before any other initialization
# WHY: Configure system behavior via environment variables


class BaseEnv:
    """Base class for environment variable handling."""
    def _init(self, name: str) -> None:
        raise NotImplementedError


T = TypeVar("T")


class EnvVar(BaseEnv, Generic[T]):
    """Type-safe environment variable with default value."""
    def __init__(self, default_value: T, fn: Callable[[str], T]):
        self.value = default_value
        self.fn = fn
        super().__init__()

    def _init(self, name: str) -> None:
        env_value = os.getenv(name)
        if env_value is not None:
            try:
                self.value = self.fn(env_value)
            except Exception:
                pass

    def __bool__(self):
        return self.value

    def __str__(self):
        return str(self.value)


_TO_BOOL = lambda x: x.lower() in ("1", "true", "yes")


def _PARSE_MEM_BYTES(mem: str) -> int:
    """Parse memory size strings like '1G', '512M', etc."""
    mem = mem.strip().upper()
    if not mem[-1].isalpha():
        return int(mem)
    if mem.endswith("B"):
        mem = mem[:-1]
    UNIT_MAP = {"K": 1024, "M": 1024**2, "G": 1024**3}
    return int(float(mem[:-1]) * UNIT_MAP[mem[-1]])


MINISGL_ENV_PREFIX = "MINISGL_"
EnvInt = partial(EnvVar[int], fn=int)
EnvFloat = partial(EnvVar[float], fn=float)
EnvBool = partial(EnvVar[bool], fn=_TO_BOOL)
EnvOption = partial(EnvVar[bool | None], fn=_TO_BOOL, default_value=None)
EnvMem = partial(EnvVar[int], fn=_PARSE_MEM_BYTES)


class EnvClassSingleton:
    """
    Singleton holding all environment configuration.

    Available env vars (prefix with MINISGL_):
      - SHELL_MAX_TOKENS: Max tokens in shell mode (default: 2048)
      - SHELL_TEMPERATURE: Temperature in shell mode (default: 0.6)
      - FLASHINFER_USE_TENSOR_CORES: Force tensor core usage
      - DISABLE_OVERLAP_SCHEDULING: Disable overlapped scheduling
      - PYNCCL_MAX_BUFFER_SIZE: Max buffer for PyNCCL (default: 1GB)
    """
    _instance: EnvClassSingleton | None = None
    SHELL_MAX_TOKENS = EnvInt(2048)
    SHELL_TEMPERATURE = EnvFloat(0.6)
    FLASHINFER_USE_TENSOR_CORES = EnvOption()
    DISABLE_OVERLAP_SCHEDULING = EnvBool(False)
    PYNCCL_MAX_BUFFER_SIZE = EnvMem(1024**3)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, BaseEnv):
                attr_value._init(f"{MINISGL_ENV_PREFIX}{attr_name}")


ENV = EnvClassSingleton()


# ==============================================================================
# Configuration Classes
# ==============================================================================
# WHEN: After args are parsed, before scheduler/engine init
# WHY: Centralized configuration for all components


@dataclass(frozen=True)
class EngineConfig:
    """
    Configuration for the inference engine.

    Used by: Engine.__init__()

    Key parameters:
      - model_path: HuggingFace model ID or local path
      - tp_info: Tensor parallelism rank/size
      - dtype: Model precision (bfloat16, float16, etc.)
      - max_running_req: Max concurrent requests
      - memory_ratio: Fraction of GPU memory for KV cache
      - cuda_graph_max_bs: Max batch size for CUDA graphs
      - attention_backend: "fa3", "fi", or "fa3,fi" (hybrid)
    """
    model_path: str
    tp_info: "DistributedInfo"
    dtype: torch.dtype = torch.bfloat16

    max_running_req: int = 64
    max_seq_len_override: int | None = None
    memory_ratio: float = 0.88
    num_page_override: int | None = None

    cuda_graph_max_bs: int | None = None
    cuda_graph_bs: List[int] | None = None

    use_dummy_weight: bool = False
    use_pynccl: bool = True
    attention_backend: str = "auto"


def _get_pid_suffix() -> str:
    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    """
    Configuration for the scheduler (extends EngineConfig).

    Additional parameters:
      - max_extend_tokens: Max tokens per prefill chunk
      - cache_type: "radix" (with prefix caching) or "naive"
      - offline_mode: True for LLM class (no network)
    """
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False

    # Networking config (ZMQ addresses)
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/minisgl_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/minisgl_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/minisgl_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True


# ##############################################################################
# PART 2: API SERVER (FastAPI)
# ##############################################################################
#
# WHEN: Runs in main process, handles HTTP requests
# WHY: Provides OpenAI-compatible REST API for inference
#
# Key components:
#   - FrontendManager: Manages async request/response correlation
#   - /v1/chat/completions: OpenAI-compatible chat API
#   - /generate: Simple text generation endpoint
#   - SSE streaming: Server-Sent Events for streaming responses


# @dataclass
# class FrontendManager:
#     """
#     Manages frontend state for async request handling.
#
#     WHEN: Created at server startup, used for every request
#
#     Key responsibilities:
#       - uid_counter: Assigns unique IDs to requests
#       - ack_map: Maps uid -> list of UserReply (streaming responses)
#       - event_map: Maps uid -> asyncio.Event (for async waiting)
#
#     Flow:
#       1. new_user() -> allocate uid, create event
#       2. send_one(TokenizeMsg) -> send to tokenizer via ZMQ
#       3. listen() -> background task receives UserReply from tokenizer
#       4. wait_for_ack(uid) -> async generator yields responses
#     """
#     config: ServerArgs
#     send_tokenizer: ZmqAsyncPushQueue  # To tokenizer
#     recv_tokenizer: ZmqAsyncPullQueue  # From tokenizer
#     uid_counter: int = 0
#     ack_map: Dict[int, List[UserReply]]   # uid -> responses
#     event_map: Dict[int, asyncio.Event]   # uid -> notify event
#
#     def new_user(self) -> int:
#         """Allocate new request ID."""
#         uid = self.uid_counter
#         self.uid_counter += 1
#         self.ack_map[uid] = []
#         self.event_map[uid] = asyncio.Event()
#         return uid
#
#     async def send_one(self, msg: TokenizeMsg):
#         """Send tokenization request to tokenizer process."""
#         await self.send_tokenizer.put(msg)
#
#     async def listen(self):
#         """Background task: receive responses and dispatch to waiters."""
#         while True:
#             msg = await self.recv_tokenizer.get()
#             for reply in unwrap_msg(msg):
#                 self.ack_map[reply.uid].append(reply)
#                 self.event_map[reply.uid].set()  # Wake up waiter
#
#     async def wait_for_ack(self, uid: int):
#         """Async generator yielding streaming responses."""
#         while True:
#             await self.event_map[uid].wait()
#             self.event_map[uid].clear()
#             pending = self.ack_map[uid]
#             self.ack_map[uid] = []
#             for ack in pending:
#                 yield ack
#                 if ack.finished:
#                     return


# HTTP Endpoint: /v1/chat/completions
#
# @app.post("/v1/chat/completions")
# async def v1_completions(req: OpenAICompletionRequest):
#     state = get_global_state()
#     uid = state.new_user()
#
#     # Send tokenization request
#     await state.send_one(TokenizeMsg(
#         uid=uid,
#         text=[msg.model_dump() for msg in req.messages],  # Chat format
#         sampling_params=SamplingParams(
#             max_tokens=req.max_tokens,
#             temperature=req.temperature,
#         ),
#     ))
#
#     # Return SSE streaming response
#     return StreamingResponse(
#         state.stream_chat_completions(uid),
#         media_type="text/event-stream",
#     )


# ##############################################################################
# PART 3: TOKENIZER WORKER (Separate Process)
# ##############################################################################
#
# WHEN: Runs in dedicated process, handles tokenization/detokenization
# WHY: Offload CPU-bound tokenization from GPU scheduler
#
# The tokenizer worker runs a loop:
#   1. Receive messages from API server (TokenizeMsg) or scheduler (DetokenizeMsg)
#   2. Process: tokenize text->ids or detokenize id->text
#   3. Send results: UserMsg to scheduler, UserReply to API server


# def tokenize_worker(tokenizer_path: str, ...):
#     """
#     Tokenizer worker process entry point.
#
#     WHEN: Spawned at server startup, runs forever
#
#     Handles two message types:
#       - TokenizeMsg (from API): text -> token IDs -> UserMsg -> scheduler
#       - DetokenizeMsg (from scheduler): token ID -> text -> UserReply -> API
#     """
#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
#     tokenize_manager = TokenizeManager(tokenizer)
#     detokenize_manager = DetokenizeManager(tokenizer)
#
#     while True:
#         pending_msg = recv_listener.get()
#
#         # Separate by message type
#         detokenize_msgs = [m for m in pending_msg if isinstance(m, DetokenizeMsg)]
#         tokenize_msgs = [m for m in pending_msg if isinstance(m, TokenizeMsg)]
#
#         # Handle detokenization (token -> text)
#         if detokenize_msgs:
#             texts = detokenize_manager.detokenize(detokenize_msgs)
#             replies = [UserReply(uid=m.uid, text=t, finished=m.finished)
#                        for m, t in zip(detokenize_msgs, texts)]
#             send_frontend.put(BatchFrontendMsg(replies))
#
#         # Handle tokenization (text -> tokens)
#         if tokenize_msgs:
#             tensors = tokenize_manager.tokenize(tokenize_msgs)
#             user_msgs = [UserMsg(uid=m.uid, input_ids=t, sampling_params=m.sampling_params)
#                          for m, t in zip(tokenize_msgs, tensors)]
#             send_backend.put(BatchBackendMsg(user_msgs))


# class TokenizeManager:
#     """Handles text -> token ID conversion."""
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer
#
#     def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
#         results = []
#         for msg in msgs:
#             if isinstance(msg.text, list):  # Chat messages
#                 text = self.tokenizer.apply_chat_template(msg.text, tokenize=False)
#             else:
#                 text = msg.text
#             ids = self.tokenizer.encode(text)
#             results.append(torch.tensor(ids, dtype=torch.int32))
#         return results


# class DetokenizeManager:
#     """
#     Handles incremental token ID -> text conversion.
#
#     Maintains state per request for proper handling of:
#       - Multi-byte UTF-8 characters
#       - Token merging (e.g., "Ġ" prefix in GPT tokenizers)
#     """
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer
#         self.decode_map: Dict[int, DecodeStatus] = {}  # uid -> state
#
#     def detokenize(self, msgs: List[DetokenizeMsg]) -> List[str]:
#         results = []
#         for msg in msgs:
#             if msg.uid not in self.decode_map:
#                 self.decode_map[msg.uid] = DecodeStatus()
#
#             status = self.decode_map[msg.uid]
#             status.all_ids.append(msg.next_token)
#
#             # Decode incrementally
#             full_text = self.tokenizer.decode(status.all_ids)
#             new_text = full_text[len(status.prefix):]
#             status.prefix = full_text
#
#             if msg.finished:
#                 del self.decode_map[msg.uid]
#
#             results.append(new_text)
#         return results


# ##############################################################################
# PART 4: CORE DATA STRUCTURES
# ##############################################################################
#
# WHEN: Used throughout the serving pipeline
# WHY: Define the fundamental abstractions for requests, batches, and context

# ==============================================================================
# Core Types (core.py)
# ==============================================================================


@dataclass
class SamplingParams:
    """
    Parameters controlling token generation.

    WHEN USED: Attached to each request, checked during sampling

    Parameters:
      - top_k: Only sample from top K tokens (1 = greedy)
      - temperature: Sampling temperature (0 = greedy)
      - max_tokens: Maximum tokens to generate
      - ignore_eos: Continue generating past EOS token
    """
    top_k: int = 1
    ignore_eos: bool = False
    temperature: float = 0.0
    max_tokens: int = 1024


class Req:
    """
    A single inference request.

    WHEN USED:
      - Created when UserMsg arrives at scheduler
      - Updated after each decode step
      - Freed when generation completes

    Key state:
      - host_ids: All token IDs (input + generated), on CPU
      - table_idx: Index into page table for this request
      - cached_len: Tokens already in KV cache (from prefix caching)
      - device_len: Total tokens on device (including new ones)

    Lifecycle:
      1. Created with input_ids from user
      2. Prefill: extend_len = device_len - cached_len tokens processed
      3. Decode: one token at a time, cached_len catches up to device_len
      4. Finished when remain_len <= 0 or EOS generated
    """
    def __init__(
        self,
        *,
        input_ids: torch.Tensor,       # CPU int32 tensor
        table_idx: int,                 # Index in page table
        cached_len: int,                # Tokens already cached
        output_len: int,                # Expected output length
        uid: int,                       # Unique request ID
        sampling_params: SamplingParams,
        cache_handle: "BaseCacheHandle",
    ) -> None:
        assert input_ids.is_cpu
        self.host_ids = input_ids
        self.table_idx = table_idx
        self.cached_len = cached_len
        self.device_len = len(input_ids)
        self.max_device_len = len(input_ids) + output_len
        self.uid = uid
        self.sampling_params = sampling_params
        self.cache_handle = cache_handle
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

    @property
    def remain_len(self) -> int:
        """Tokens left to generate."""
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        """Tokens to process this step (not yet in cache)."""
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        """Called after each decode step."""
        self.cached_len = self.device_len
        self.device_len += 1

    def append_host(self, next_token: torch.Tensor) -> None:
        """Append generated token to host_ids."""
        self.host_ids = torch.cat([self.host_ids, next_token])

    def can_decode(self) -> bool:
        """Check if request can continue to decode phase."""
        return self.remain_len > 0


class Batch:
    """
    A batch of requests for parallel inference.

    WHEN USED:
      - Created by PrefillManager.schedule_next_batch() or
        DecodeManager.schedule_next_batch()
      - Passed to Engine.forward_batch()

    Two phases:
      - "prefill": Processing initial tokens (variable length per request)
      - "decode": Generating one token per request (all same length)

    Key attributes set by scheduler:
      - input_ids: Flattened token IDs for all requests
      - out_loc: KV cache indices to write new K/V values
      - attn_metadata: Backend-specific attention metadata
    """
    def __init__(self, *, reqs: List[Req], phase: Literal["prefill", "decode"]):
        self.reqs = reqs
        self.phase: Literal["prefill", "decode"] = phase
        # Set by scheduler:
        self.input_ids: torch.Tensor
        self.out_loc: torch.Tensor
        self.padded_reqs: List[Req]  # May include padding for CUDA graphs
        # Set by attention backend:
        self.attn_metadata: "BaseAttnMetadata"

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        return len(self.padded_reqs)


class Context:
    """
    Global inference context (singleton per GPU).

    WHEN USED:
      - Created by Engine.__init__()
      - Set as global via set_global_ctx()
      - Accessed during forward pass via get_global_ctx()

    Holds:
      - kv_cache: The KV cache storage
      - attn_backend: Attention implementation (FA3/FlashInfer)
      - page_table: Maps (request, position) -> cache page
      - batch: Current batch being processed (set via context manager)
    """
    def __init__(
        self,
        *,
        page_size: int,
        kv_cache: "BaseKVCache",
        attn_backend: "BaseAttnBackend",
        page_table: torch.Tensor,
    ):
        self._batch: Batch | None = None
        self.page_table = page_table
        assert (
            self.page_table.dim() == 2
            and self.page_table.is_cuda
            and self.page_table.dtype == torch.int32
            and self.page_table.is_contiguous()
        )
        self.kv_cache = kv_cache
        self.attn_backend = attn_backend
        assert page_size == 1  # Only page_size=1 supported

    def set_batch(self, batch: Batch):
        assert self._batch is None
        self._batch = batch

    def reset_batch(self):
        assert self._batch is not None
        self._batch = None

    @contextmanager
    def forward_batch(self, batch: Batch):
        """Context manager to set current batch during forward pass."""
        self.set_batch(batch)
        try:
            yield
        finally:
            self.reset_batch()

    @property
    def batch(self) -> Batch:
        assert self._batch is not None, "Global batch is not set"
        return self._batch


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    """Set the global context (called by Engine.__init__)."""
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    """Get the global context (called during forward pass)."""
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX


# ==============================================================================
# Message Protocol (message/*.py)
# ==============================================================================
# WHEN: For inter-process communication (Scheduler <-> Tokenizer <-> API Server)
# WHY: Serialize requests/responses across process boundaries via ZMQ


@dataclass
class BaseBackendMsg:
    """
    Base class for messages TO the scheduler (backend).

    Sent by: Tokenizer process
    Received by: Scheduler process
    """
    pass


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    """Batch of backend messages."""
    data: List[BaseBackendMsg]


@dataclass
class ExitMsg(BaseBackendMsg):
    """Signal scheduler to shutdown."""
    pass


@dataclass
class UserMsg(BaseBackendMsg):
    """
    New inference request from user.

    WHEN: After tokenization, sent to scheduler

    Fields:
      - uid: Unique request ID (for correlation)
      - input_ids: Tokenized prompt (CPU int32)
      - sampling_params: Generation parameters
    """
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams


@dataclass
class BaseFrontendMsg:
    """Base class for messages TO the API server (frontend)."""
    pass


@dataclass
class BatchFrontendMsg(BaseFrontendMsg):
    """Batch of frontend messages."""
    data: List[BaseFrontendMsg]


@dataclass
class UserReply(BaseFrontendMsg):
    """
    Streaming response to user.

    WHEN: After detokenization, sent to API server

    Fields:
      - uid: Request ID (matches UserMsg.uid)
      - incremental_output: New text since last reply
      - finished: True if generation complete
    """
    uid: int
    incremental_output: str
    finished: bool


@dataclass
class BaseTokenizerMsg:
    """Base class for messages TO/FROM tokenizer."""
    pass


@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    """Batch of tokenizer messages."""
    data: List[BaseTokenizerMsg]


@dataclass
class DetokenizeMsg(BaseTokenizerMsg):
    """
    Request to convert token ID to text.

    WHEN: After each decode step, sent from scheduler to tokenizer
    """
    uid: int
    next_token: int
    finished: bool


@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    """
    Request to convert text to token IDs.

    WHEN: When new request arrives from API server
    """
    uid: int
    text: str | List[Dict[str, str]]  # String or chat messages
    sampling_params: SamplingParams


# ##############################################################################
# PART 5: DISTRIBUTED COMMUNICATION
# ##############################################################################
#
# WHEN: At startup (set_tp_info) and during forward pass (all_reduce/all_gather)
# WHY: Enable tensor parallelism across multiple GPUs


# ==============================================================================
# DistributedInfo (distributed/info.py)
# ==============================================================================


@dataclass(frozen=True)
class DistributedInfo:
    """
    Tensor parallelism rank and world size.

    WHEN SET: Early in process startup, before model creation
    WHY: All TP-aware layers need to know their shard

    Example: For 4-GPU TP, each process has rank in [0,1,2,3] and size=4
    """
    rank: int
    size: int

    def __post_init__(self):
        assert 0 <= self.rank < self.size

    def is_primary(self) -> bool:
        """Only rank 0 handles I/O with tokenizer."""
        return self.rank == 0


_TP_INFO: DistributedInfo | None = None


def set_tp_info(rank: int, size: int) -> None:
    """Set TP info (called early in each process)."""
    global _TP_INFO
    if _TP_INFO is not None:
        raise RuntimeError("TP info has been set")
    _TP_INFO = DistributedInfo(rank, size)


def get_tp_info() -> DistributedInfo:
    """Get TP info (called by all TP-aware layers)."""
    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO


def try_get_tp_info() -> DistributedInfo | None:
    """Get TP info if set, else None."""
    return _TP_INFO


# ==============================================================================
# Distributed Implementation (distributed/impl.py)
# ==============================================================================


@dataclass
class DistributedImpl(ABC):
    """Abstract base for distributed communication backends."""

    @abstractmethod
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Sum tensor across all ranks."""
        ...

    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        """Gather tensor from all ranks."""
        ...


@dataclass
class TorchDistributedImpl(DistributedImpl):
    """PyTorch native distributed backend (fallback)."""

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        shape = list(x.shape)
        shape[0] = shape[0] * tp_size
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x)
        return out


@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    """
    PyNCCL-based distributed backend (preferred).

    WHY: Better performance than torch.distributed for custom collectives
    """
    comm: "PyNCCLCommunicator"

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        self.comm.all_reduce(x, "sum")
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        world_size = get_tp_info().size
        output_shape = list(x.shape)
        output_shape[0] *= world_size
        result = x.new_empty(output_shape)
        self.comm.all_gather(result, x)
        return result


class DistributedCommunicator:
    """
    Singleton communicator with pluggable backends.

    WHEN USED:
      - LinearOProj.forward() -> all_reduce after row-parallel matmul
      - LinearRowParallel.forward() -> all_reduce
      - VocabParallelEmbedding.forward() -> all_reduce
      - ParallelLMHead.forward() -> all_gather for logits
    """
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_gather(x)


def enable_pynccl_distributed(
    tp_info: DistributedInfo, tp_cpu_group, max_bytes: int
) -> None:
    """Enable PyNCCL backend (called during Engine init if use_pynccl=True)."""
    if tp_info.size == 1:
        return
    from minisgl.kernel import init_pynccl
    comm = init_pynccl(
        tp_rank=tp_info.rank,
        tp_size=tp_info.size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_bytes,
    )
    DistributedCommunicator.plugins.append(PyNCCLDistributedImpl(comm))


def destroy_distributed() -> None:
    """Cleanup distributed backends."""
    DistributedCommunicator.plugins = []


# ##############################################################################
# PART 6: SCHEDULER (The Brain)
# ##############################################################################
#
# WHEN: Runs continuously in scheduler process
# WHY: Orchestrates which requests to process and when


# ==============================================================================
# TableManager (scheduler/table.py)
# ==============================================================================
# WHEN: Allocates slots in page table for new requests


class TableManager:
    """
    Manages request slots in the page table.

    Each request gets a row in page_table (up to max_running_reqs).
    The row maps token positions to KV cache pages.
    """
    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        self._max_running_reqs = max_running_reqs
        self._free_slots = list(range(max_running_reqs))
        self.page_table = page_table
        self.token_pool = torch.empty_like(page_table, dtype=torch.int32)

    @property
    def available_size(self) -> int:
        return len(self._free_slots)

    def allocate(self) -> int:
        """Allocate a table slot for a new request."""
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        """Free a table slot when request completes."""
        self._free_slots.append(slot)


# ==============================================================================
# DecodeManager (scheduler/decode.py)
# ==============================================================================


@dataclass
class DecodeManager:
    """
    Manages requests in the decode phase.

    WHEN USED: After prefill completes, requests move here for autoregressive generation.

    Each decode step:
      1. schedule_next_batch() returns all running requests
      2. Engine generates one token per request
      3. Finished requests are removed
    """
    running_reqs: Set[Req] = field(default_factory=set)

    def add_reqs(self, reqs) -> None:
        """Add requests that can continue decoding."""
        self.running_reqs.update(req for req in reqs if req.can_decode())

    def remove_req(self, req: Req) -> None:
        """Remove finished request."""
        self.running_reqs.discard(req)

    @property
    def inflight_tokens(self) -> int:
        """Total tokens being generated (for memory estimation)."""
        return sum(req.remain_len for req in self.running_reqs)

    def schedule_next_batch(self) -> Batch | None:
        """Create decode batch with all running requests."""
        if not self.runnable:
            return None
        return Batch(reqs=list(self.running_reqs), phase="decode")

    @property
    def runnable(self) -> bool:
        return bool(self.running_reqs)


# ==============================================================================
# PrefillManager (scheduler/prefill.py)
# ==============================================================================


@dataclass
class PendingReq:
    """A request waiting for prefill."""
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    chunked_req: "ChunkedReq | Req | None" = None

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens


class ChunkedReq(Req):
    """
    A partially-prefilled request.

    For long prompts, prefill is split into chunks to avoid OOM.
    ChunkedReq is not yet ready for decode (can_decode returns False).
    """
    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedReq should not be sampled")

    def can_decode(self) -> bool:
        return False


@dataclass
class PrefillAdder:
    """
    Helper to add requests to a prefill batch.

    Respects token budget and memory constraints.
    """
    token_budget: int
    reserved_size: int
    cache_manager: "CacheManager"
    table_manager: TableManager

    def _try_allocate_one(self, req: PendingReq) -> Tuple["BaseCacheHandle", int] | None:
        """Try to allocate resources for one request."""
        if self.table_manager.available_size == 0:
            return None

        # Check prefix cache
        handle, match_indices = self.cache_manager.match_req(req)
        cached_len = handle.cached_len
        extend_len = req.input_len - cached_len
        estimated_len = extend_len + req.output_len

        # Check memory
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None

        self.cache_manager.lock(handle)
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            self.cache_manager.unlock(handle)
            return None

        table_idx = self.table_manager.allocate()
        if cached_len > 0:
            # Copy cached indices to page table
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
            page_entry.copy_(match_indices)

        return handle, table_idx

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        """Try to add one request to the batch."""
        if self.token_budget <= 0:
            return None

        if chunked_req := pending_req.chunked_req:
            # Continue chunked prefill
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
            )

        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx = resource
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
            )

        return None

    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: "BaseCacheHandle",
        table_idx: int,
        cached_len: int,
    ) -> Req:
        """Create Req (or ChunkedReq) for the batch."""
        remain_len = pending_req.input_len - cached_len
        chunk_size = min(self.token_budget, remain_len)
        is_chunked = chunk_size < remain_len
        CLS = ChunkedReq if is_chunked else Req
        self.token_budget -= chunk_size
        self.reserved_size += remain_len + pending_req.output_len

        # Copy token IDs to device
        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx][_slice]
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)

        return CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
        )


@dataclass
class PrefillManager:
    """
    Manages requests waiting for prefill.

    WHEN USED: New requests go here first, then move to DecodeManager.

    Key method: schedule_next_batch()
      - Tries to add as many pending requests as fit in token budget
      - Uses prefix caching to skip already-cached tokens
      - May chunk long requests across multiple batches
    """
    cache_manager: "CacheManager"
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: List[PendingReq] = field(default_factory=list)

    def add_one_req(self, req: UserMsg) -> None:
        """Add new request from user."""
        self.pending_list.append(PendingReq(req.uid, req.input_ids, req.sampling_params))

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        """Create prefill batch with pending requests."""
        if len(self.pending_list) == 0:
            return None

        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )

        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []

        for pending_req in self.pending_list:
            if req := adder.try_add_one(pending_req):
                pending_req.chunked_req = None
                if isinstance(req, ChunkedReq):
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
            else:
                break  # Can't fit more

        if len(reqs) == 0:
            return None

        self.pending_list = chunked_list + self.pending_list[len(reqs):]
        return Batch(reqs=reqs, phase="prefill")

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0


# ==============================================================================
# CacheManager (scheduler/cache.py)
# ==============================================================================


class CacheManager:
    """
    Manages KV cache allocation with prefix caching.

    WHEN USED:
      - match_req(): Find cached prefix for new request
      - allocate(): Get cache pages for new tokens
      - free_and_cache_finished_req(): Return pages and update radix tree

    Two-level structure:
      1. Free slots: Available cache pages
      2. Radix tree (via manager): Cached prefixes that can be evicted
    """
    def __init__(self, device: torch.device, num_pages: int, type: str):
        self._free_slots = torch.arange(num_pages, dtype=torch.int32, device=device)
        self.device = device
        # Create radix or naive manager
        from minisgl.kvcache import create_cache_manager
        self.manager = create_cache_manager(device=device, type=type)
        self.num_pages = num_pages

    def _free(self, indices: torch.Tensor) -> None:
        """Return pages to free pool."""
        if len(indices) > 0:
            self._free_slots = torch.cat([self._free_slots, indices])

    def match_req(self, req: PendingReq):
        """Find longest cached prefix for request."""
        input_len = req.input_len
        assert input_len > 0
        return self.manager.match_prefix(req.input_ids[: input_len - 1])

    @property
    def available_size(self) -> int:
        """Total available pages (free + evictable)."""
        return self.manager.size_info.evictable_size + len(self._free_slots)

    def lock(self, handle: "BaseCacheHandle") -> None:
        """Protect cached prefix from eviction."""
        self.manager.lock_handle(handle, unlock=False)

    def unlock(self, handle: "BaseCacheHandle") -> None:
        """Allow cached prefix to be evicted."""
        self.manager.lock_handle(handle, unlock=True)

    def allocate(self, needed_len: int) -> torch.Tensor:
        """Allocate cache pages (may evict old prefixes)."""
        if needed_len <= (free_len := len(self._free_slots)):
            allocated = self._free_slots[:needed_len]
            self._free_slots = self._free_slots[needed_len:]
            return allocated

        # Need to evict
        evicted = self.manager.evict(needed_len - free_len)
        merged = torch.cat([self._free_slots, evicted])
        assert len(merged) >= needed_len

        allocated = merged[:needed_len]
        self._free_slots = merged[needed_len:]
        return allocated

    def free_and_cache_finished_req(
        self,
        old_handle: "BaseCacheHandle",
        input_ids: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        """Cache finished request's prefix and free remaining pages."""
        in_cache_len = self.manager.insert_prefix(input_ids, indices)
        self._free(indices[old_handle.cached_len : in_cache_len])
        self.unlock(old_handle)

    def check_integrity(self) -> None:
        """Verify cache state is consistent."""
        self.manager.check_integrity()
        if len(self._free_slots) + self.manager.size_info.total_size != self.num_pages:
            raise RuntimeError("CacheManager integrity check failed")


# ==============================================================================
# Scheduler Main Class (scheduler/scheduler.py)
# ==============================================================================


class ForwardInput(NamedTuple):
    """Input prepared for a forward pass."""
    batch: Batch
    sample_args: "BatchSamplingArgs"
    load_indices: torch.Tensor
    write_indices: torch.Tensor


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler:
    """
    Main scheduler class - the brain of the inference server.

    WHEN USED: Runs continuously in scheduler process

    Main loop (run_forever):
      1. receive_msg(): Get new requests from tokenizer
      2. _schedule_next_batch(): Choose prefill or decode
      3. _prepare_batch(): Allocate cache, prepare metadata
      4. _forward(): Execute model (on engine stream)
      5. _process_last_data(): Handle results, send to tokenizer

    Supports overlap scheduling: CPU processing overlaps with GPU compute.
    """
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.engine = Engine(config)
        self.device = self.engine.device

        # Use separate stream for scheduling (overlap with engine)
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # Initialize managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(self.device, self.engine.num_pages, config.cache_type)
        self.decode_manager = DecodeManager()
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        self.tp_info = config.tp_info
        self.finished_reqs: Set[Req] = set()

        # Load tokenizer for EOS detection
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id

        self.page_table = self.engine.page_table
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens

    def _schedule_next_batch(self) -> ForwardInput | None:
        """Choose next batch (prefill has priority, then decode)."""
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        return self._prepare_batch(batch) if batch else None

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        """Allocate cache and prepare metadata for batch."""
        needed_size = sum(r.extend_len for r in batch.reqs)
        batch.out_loc = self.cache_manager.allocate(needed_size)

        # Pad for CUDA graph
        if padding_size := self.engine.graph_runner.pad_batch(batch):
            batch.out_loc = F.pad(batch.out_loc, (0, padding_size), value=self.engine.dummy_page)

        # Prepare indices for loading/writing token IDs
        load_indices = self._make_2d_indices(
            self.token_pool,
            [(r.table_idx, r.cached_len, r.device_len) for r in batch.padded_reqs]
        )
        write_indices = self._make_2d_indices(
            self.token_pool,
            [(r.table_idx, r.device_len, r.device_len + 1) for r in batch.reqs]
        )

        # Update page table and prepare attention metadata
        self.page_table.view(-1)[load_indices] = batch.out_loc
        self.engine.attn_backend.prepare_metadata(batch)

        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            load_indices=load_indices,
            write_indices=write_indices,
        )

    def _make_2d_indices(self, table_2d: torch.Tensor, ranges: List[Tuple[int, int, int]]) -> torch.Tensor:
        """Convert 2D ranges to 1D indices for page table access."""
        assert table_2d.dim() == 2 and table_2d.is_contiguous()
        STRIDE = table_2d.stride(0)
        needed_size = sum(end - begin for _, begin, end in ranges)
        indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
        offset = 0
        for entry, begin, end in ranges:
            length = end - begin
            offset += length
            torch.arange(
                begin + entry * STRIDE,
                end + entry * STRIDE,
                dtype=torch.int32,
                out=indices_host[offset - length : offset],
            )
        return indices_host.to(table_2d.device, non_blocking=True)

    def _forward(self, forward_input: ForwardInput) -> "ForwardOutput":
        """Execute forward pass on engine."""
        # Load token IDs
        batch = forward_input.batch
        batch.input_ids = self.token_pool.view(-1)[forward_input.load_indices]

        # Run model
        forward_output = self.engine.forward_batch(batch, forward_input.sample_args)

        # Write next tokens back
        self.token_pool.view(-1)[forward_input.write_indices] = forward_output.next_tokens_gpu

        # Move successful requests to decode
        self.decode_manager.add_reqs(forward_input.batch.reqs)

        return forward_output

    def _process_last_data(
        self, last_data: ForwardData | None, ongoing_data: ForwardData | None
    ) -> None:
        """Process results from previous batch, send to tokenizer."""
        if last_data is None:
            return

        batch = last_data[0].batch
        _, next_tokens_cpu, copy_done = last_data[1]
        copy_done.synchronize()

        reply = BatchTokenizerMsg(data=[])

        for i, req in enumerate(batch.reqs):
            if req in self.finished_reqs or isinstance(req, ChunkedReq):
                continue

            next_token_id = next_tokens_cpu[i]
            req.append_host(next_token_id.unsqueeze(0))
            next_token = int(next_token_id.item())

            # Check finish conditions
            finished = req.remain_len <= 0
            if not req.sampling_params.ignore_eos:
                finished |= next_token == self.eos_token_id
            if req.device_len >= self.engine.max_seq_len - 1:
                finished = True

            reply.data.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

            if finished:
                self.finished_reqs.add(req)
                self.decode_manager.remove_req(req)

        # Free resources for finished requests
        ongoing_reqs = ongoing_data[0].batch.reqs if ongoing_data else []
        for req in self.finished_reqs.difference(ongoing_reqs):
            self.table_manager.free(req.table_idx)
            self.cache_manager.free_and_cache_finished_req(
                req.cache_handle,
                req.host_ids[: req.cached_len],
                self.page_table[req.table_idx, : req.cached_len],
            )

        self.finished_reqs.intersection_update(ongoing_reqs)
        # Send results to tokenizer (via ZMQ)
        # self.send_result(reply)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        """
        Main scheduling loop.

        Supports two modes:
          - Normal: Sequential scheduling and execution
          - Overlap: CPU scheduling overlaps with GPU execution
        """
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            # Normal mode
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self._normal_loop()
        else:
            # Overlap mode
            data = None
            while True:
                data = self._overlap_loop(data)

    def _normal_loop(self) -> None:
        """Simple sequential loop."""
        # Receive messages
        # for msg in self.receive_msg(blocking=not runnable): ...

        forward_input = self._schedule_next_batch()
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))
            self._process_last_data(ongoing_data, None)

    def _overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """Overlapped loop for better GPU utilization."""
        forward_input = self._schedule_next_batch()
        ongoing_data = None

        if forward_input is not None:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data, ongoing_data)
        return ongoing_data


# ##############################################################################
# PART 7: ENGINE (The Executor)
# ##############################################################################
#
# WHEN: Called by scheduler for each batch
# WHY: Manages model execution, CUDA graphs, and sampling


# ==============================================================================
# Sampler (engine/sample.py)
# ==============================================================================


class BatchSamplingArgs:
    """Arguments for sampling a batch."""
    def __init__(self, temperatures: torch.Tensor | None):
        self.temperatures = temperatures


class Sampler:
    """
    Token sampler.

    WHEN USED: After model forward, converts logits to next tokens.

    Supports:
      - Greedy (temperature=0): argmax
      - Temperature sampling: softmax + multinomial
    """
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        """Prepare sampling args before forward pass."""
        if all(r.sampling_params.temperature <= 0.0 for r in batch.reqs):
            return BatchSamplingArgs(temperatures=None)
        MIN_T = 1e-5
        return BatchSamplingArgs(
            temperatures=torch.tensor(
                [max(r.sampling_params.temperature, MIN_T) for r in batch.reqs],
                dtype=torch.float32, pin_memory=True,
            ).to(self.device, non_blocking=True)
        )

    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        """Sample next tokens from logits."""
        with nvtx.range("Sampler"):
            if args.temperatures is None:
                # Greedy
                return torch.argmax(logits, dim=-1)
            return self._sample_with_temperature(logits, args.temperatures)

    def _sample_with_temperature(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        """Temperature-based sampling."""
        logits.div_(temperatures.unsqueeze(-1))
        torch.softmax(logits, dim=-1, out=logits)
        return torch.multinomial(logits, num_samples=1).view(-1)


# ==============================================================================
# ForwardOutput
# ==============================================================================


class ForwardOutput(NamedTuple):
    """Output from a forward pass."""
    next_tokens_gpu: torch.Tensor    # On GPU
    next_tokens_cpu: torch.Tensor    # Copied to CPU
    copy_done_event: torch.cuda.Event  # Signals when copy is complete


# ==============================================================================
# Engine (engine/engine.py) - Simplified
# ==============================================================================
#
# The full Engine class handles:
#   1. Device initialization and TP setup
#   2. Model loading (create_model + load_hf_weight)
#   3. KV cache allocation (create_kvcache)
#   4. Attention backend setup (create_attention_backend)
#   5. CUDA graph capture (GraphRunner)
#   6. Forward pass execution
#
# Key method:
#
# def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
#     logits = self.model.forward_batch(batch)  # Run model
#     next_tokens = self.sampler.sample(logits, args)  # Sample
#     # Async copy to CPU
#     next_tokens_cpu = next_tokens.to("cpu", non_blocking=True)
#     event = torch.cuda.Event()
#     event.record()
#     return ForwardOutput(next_tokens, next_tokens_cpu, event)


# ##############################################################################
# PART 8: MODELS
# ##############################################################################
#
# WHEN: Created during Engine init, called during forward pass
# WHY: Define the neural network architecture


# ==============================================================================
# Model Config (models/config.py)
# ==============================================================================


@dataclass(frozen=True)
class RotaryConfig:
    """RoPE (Rotary Position Embedding) configuration."""
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, float] | None


@dataclass(frozen=True)
class ModelConfig:
    """
    Model architecture configuration.

    WHEN: Created from HuggingFace config during model loading.
    WHY: Standardized config format for model construction.
    """
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool

    @classmethod
    def from_hf(cls, config) -> "ModelConfig":
        """Create from HuggingFace AutoConfig."""
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=head_dim,
                max_position=config.max_position_embeddings,
                base=config.rope_theta,
                scaling=getattr(config, "rope_scaling", None),
            ),
        )


# ==============================================================================
# Base Model Class (models/base.py)
# ==============================================================================


class BaseLLMModel(ABC):
    """
    Base class for all LLM models.

    WHEN: Subclassed by Llama/Qwen3
    WHY: Standard interface for model forward pass
    """
    @abstractmethod
    def forward(self) -> torch.Tensor:
        """Run forward pass, return logits."""
        ...

    def forward_batch(self, batch: Batch) -> torch.Tensor:
        """Forward with batch context."""
        ctx = get_global_ctx()
        with ctx.forward_batch(batch):
            return self.forward()


# ==============================================================================
# Qwen3 Model (models/qwen3.py)
# ==============================================================================
#
# EXECUTION FLOW during forward():
#
#   input_ids (bs, seq_len)
#        │
#        ▼
#   VocabParallelEmbedding ──────────────────────────────────────────►
#        │                                                            │
#        ▼  hidden (bs*seq_len, hidden_size)                         │
#   ┌─────────────────────────────────────────────────────────────┐  │
#   │  For each decoder layer (num_layers times):                  │  │
#   │                                                              │  │
#   │    RMSNormFused (input_layernorm)                           │  │
#   │        │                                                     │  │
#   │        ▼                                                     │  │
#   │    RopeAttn.forward()                                       │  │
#   │        ├── LinearQKVMerged → Q, K, V                        │  │
#   │        ├── RMSNorm on Q, K (Qwen3 has QK-norm)             │  │
#   │        ├── AttentionLayer                                   │  │
#   │        │     ├── RoPE (rotary embeddings)                   │  │
#   │        │     ├── FlashInfer/FA3 attention                   │  │
#   │        │     └── store_kv (save to cache)                   │  │
#   │        └── LinearOProj → attention output                   │  │
#   │        │                                                     │  │
#   │    RMSNormFused (post_attention_layernorm)                  │  │
#   │        │                                                     │  │
#   │        ▼                                                     │  │
#   │    GatedMLP.forward()                                       │  │
#   │        ├── LinearColParallelMerged (gate_up_proj)           │  │
#   │        ├── silu_and_mul (SiLU activation with gating)       │  │
#   │        └── LinearRowParallel (down_proj)                    │  │
#   │                                                              │  │
#   └──────────────────────────────────────────────────────────────┘  │
#        │                                                            │
#        ▼  hidden (bs*seq_len, hidden_size)                         │
#   RMSNormFused (final norm)                                        │
#        │                                                            │
#        ▼                                                            │
#   ParallelLMHead ◄─────────────(tied weights if configured)────────┘
#        │
#        ▼
#   logits (bs, vocab_size)


class Qwen3DecoderLayer:
    """
    Single Qwen3 transformer decoder layer.

    DIFFERENCE FROM LLAMA: Has QK-norm (RMSNorm on Q and K before attention)
    """
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = RopeAttn(config, layer_id, has_qk_norm=True)  # QK-norm!
        self.mlp = GatedMLP(config)
        self.input_layernorm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self._layer_id = layer_id

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-attention norm + residual
        x, residual = self.input_layernorm.forward(x, residual)

        # Self-attention
        with nvtx.range(f"MHA_{self._layer_id}"):
            x = self.self_attn.forward(x)

        # Post-attention norm + residual
        x, residual = self.post_attention_layernorm.forward(x, residual)

        # FFN
        with nvtx.range(f"MLP_{self._layer_id}"):
            x = self.mlp.forward(x)

        return x, residual


class Qwen3Model:
    """Qwen3 transformer (without LM head)."""
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = [Qwen3DecoderLayer(config, i) for i in range(config.num_layers)]
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Embedding
        with nvtx.range("Embedding"):
            x = self.embed_tokens.forward(input_ids)

        # Transformer layers
        residual: torch.Tensor | None = None
        for layer in self.layers:
            with nvtx.range(f"Layer_{layer._layer_id}"):
                x, residual = layer.forward(x, residual)

        # Final norm
        return self.norm.forward(x, residual)[0]


class Qwen3ForCausalLM(BaseLLMModel):
    """
    Qwen3 model with causal LM head.

    WHEN: Created by create_model() when model_path contains "qwen3"
    """
    def __init__(self, config: ModelConfig):
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )

    def forward(self) -> torch.Tensor:
        ctx = get_global_ctx()
        output = self.model.forward(ctx.batch.input_ids)
        with nvtx.range("LMHead"):
            logits = self.lm_head.forward(output)
        return logits


# ==============================================================================
# Llama Model (models/llama.py) - Nearly identical to Qwen3
# ==============================================================================


class LlamaDecoderLayer:
    """Single Llama decoder layer (no QK-norm)."""
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = RopeAttn(config, layer_id)  # No QK-norm
        self.mlp = GatedMLP(config)
        self.input_layernorm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self._layer_id = layer_id

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        with nvtx.range(f"MHA_{self._layer_id}"):
            x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        with nvtx.range(f"MLP_{self._layer_id}"):
            x = self.mlp.forward(x)
        return x, residual


class LlamaModel:
    """Llama transformer (without LM head)."""
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = [LlamaDecoderLayer(config, i) for i in range(config.num_layers)]
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        with nvtx.range("Embedding"):
            x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers:
            with nvtx.range(f"Layer_{layer._layer_id}"):
                x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class LlamaForCausalLM(BaseLLMModel):
    """Llama model with causal LM head."""
    def __init__(self, config: ModelConfig):
        self.model = LlamaModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )

    def forward(self) -> torch.Tensor:
        ctx = get_global_ctx()
        output = self.model.forward(ctx.batch.input_ids)
        with nvtx.range("LMHead"):
            logits = self.lm_head.forward(output)
        return logits


def create_model(model_path: str, model_config: ModelConfig) -> BaseLLMModel:
    """Factory function to create model based on path."""
    model_name = model_path.lower()
    if "llama" in model_name:
        return LlamaForCausalLM(model_config)
    elif "qwen3" in model_name:
        return Qwen3ForCausalLM(model_config)
    else:
        raise ValueError(f"Unsupported model: {model_path}")


# ==============================================================================
# Model Utilities: GatedMLP, RopeAttn (models/utils.py)
# ==============================================================================


class GatedMLP:
    """
    Gated MLP (SwiGLU) used in Llama/Qwen.

    WHEN: Called in each decoder layer after attention.

    Architecture:
      hidden -> gate_up_proj -> [gate, up] -> SiLU(gate) * up -> down_proj -> hidden
    """
    def __init__(self, config: ModelConfig):
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            has_bias=False,
        )
        match config.hidden_act:
            case "silu":
                self.act_fn = silu_and_mul
            case act_fn:
                raise ValueError(f"Unsupported activation: {act_fn}")
        self.down_proj = LinearRowParallel(
            config.intermediate_size,
            config.hidden_size,
            has_bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward(x)
        del x
        y = self.act_fn(gate_up)
        del gate_up
        return self.down_proj.forward(y)


class RopeAttn:
    """
    Self-attention with RoPE.

    WHEN: Called in each decoder layer.

    Architecture:
      hidden -> qkv_proj -> Q, K, V -> [optional QK-norm] ->
      AttentionLayer (RoPE + attention + KV store) -> o_proj -> hidden
    """
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        has_attn_bias: bool = False,
        has_qk_norm: bool = False,  # Qwen3 has this, Llama doesn't
    ):
        head_dim = config.head_dim
        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=has_attn_bias,
        )
        self.has_qk_norm = has_qk_norm
        if has_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None
        self.attn = AttentionLayer(
            layer_id=layer_id,
            head_dim=head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_config=config.rotary_config,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )
        self.o_proj = LinearOProj(
            head_dim * config.num_qo_heads,
            config.hidden_size,
            has_bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj.forward(x)
        del x
        o = self.attn.forward(qkv)
        return self.o_proj.forward(o)


# ##############################################################################
# PART 9: LAYERS (Building Blocks)
# ##############################################################################
#
# WHEN: Used by model classes during forward pass
# WHY: Reusable components for transformer architecture


# ==============================================================================
# Base Layer Classes (layers/base.py)
# ==============================================================================

_STATE_DICT: TypeAlias = Dict[str, torch.Tensor]


def _concat_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


class BaseOP:
    """
    Base class for all operations with learnable parameters.

    Provides:
      - state_dict(): Collect all tensor parameters
      - load_state_dict(): Load parameters from dict
    """
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        result = result if result is not None else {}
        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                result[_concat_prefix(prefix, name)] = param
            elif isinstance(param, BaseOP):
                param.state_dict(prefix=_concat_prefix(prefix, name), result=result)
        return result

    def load_state_dict(self, state_dict: _STATE_DICT, *, prefix: str = "", _internal: bool = False) -> None:
        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                item = state_dict.pop(_concat_prefix(prefix, name))
                assert param.shape == item.shape and param.dtype == item.dtype
                setattr(self, name, item)
            elif isinstance(param, BaseOP):
                param.load_state_dict(state_dict, prefix=_concat_prefix(prefix, name), _internal=True)
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys: {list(state_dict.keys())}")


class StateLessOP(BaseOP):
    """Base class for operations without learnable parameters (e.g., RoPE)."""
    def __init__(self):
        super().__init__()

    def load_state_dict(self, state_dict: _STATE_DICT, *, prefix: str = "", _internal: bool = False) -> None:
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys: {list(state_dict.keys())}")

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        return result if result is not None else {}


# ==============================================================================
# Linear Layers with Tensor Parallelism (layers/linear.py)
# ==============================================================================
#
# Tensor Parallelism Strategy:
#
#   Column Parallel (LinearColParallelMerged, LinearQKVMerged):
#     - Weight sharded along output dim
#     - Each rank computes output_size/tp_size features
#     - No communication needed during forward
#
#   Row Parallel (LinearRowParallel, LinearOProj):
#     - Weight sharded along input dim
#     - Each rank computes full output from its shard
#     - Requires all_reduce to sum partial results


class _LinearTPImpl(BaseOP):
    """Base linear layer with TP support."""
    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
    ):
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self.weight = torch.empty(local_osize, local_isize)
        self.bias = torch.empty(local_osize) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class LinearColParallelMerged(_LinearTPImpl):
    """
    Column-parallel linear with merged outputs.

    WHEN: gate_up_proj in GatedMLP
    WHY: Compute gate and up projections together, sharded across ranks
    """
    def __init__(self, input_size: int, output_sizes: List[int], has_bias: bool):
        tp_info = get_tp_info()
        tp_output_sizes = [divide_even(size, tp_info.size) for size in output_sizes]
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)
        super().__init__(input_size, output_size, input_size, tp_output_size, has_bias)


class LinearQKVMerged(_LinearTPImpl):
    """
    Merged Q, K, V projections.

    WHEN: qkv_proj in RopeAttn
    WHY: Single matmul for all three projections
    """
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()
        GQA_ratio = divide_even(num_qo_heads, num_kv_heads)
        local_num_kv = divide_even(num_kv_heads, tp_info.size)
        full_isize = hidden_size
        full_osize = (GQA_ratio + 2) * num_kv_heads * head_dim
        local_isize = hidden_size
        local_osize = (GQA_ratio + 2) * local_num_kv * head_dim
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)


class LinearOProj(_LinearTPImpl):
    """
    Output projection (row-parallel).

    WHEN: o_proj in RopeAttn
    WHY: Project attention output back to hidden_size, with all_reduce
    """
    def __init__(self, input_size: int, output_size: int, has_bias: bool):
        tp_info = get_tp_info()
        local_isize = divide_even(input_size, tp_info.size)
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(input_size, output_size, local_isize, output_size, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class LinearRowParallel(_LinearTPImpl):
    """
    Row-parallel linear.

    WHEN: down_proj in GatedMLP
    WHY: Input is sharded, output needs all_reduce
    """
    def __init__(self, input_size: int, output_size: int, has_bias: bool):
        tp_info = get_tp_info()
        local_input_size = divide_even(input_size, tp_info.size)
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(input_size, output_size, local_input_size, output_size, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


# ==============================================================================
# Embeddings (layers/embedding.py)
# ==============================================================================


class VocabParallelEmbedding(BaseOP):
    """
    Vocabulary-parallel embedding.

    WHEN: embed_tokens in model
    WHY: Large vocab split across TP ranks

    Each rank holds vocab_size/tp_size embeddings.
    For OOV tokens, returns zeros, then all_reduce sums them.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        tp_info = get_tp_info()
        tp_rank = tp_info.rank
        self.tp_size = tp_info.size
        self.num_embeddings = num_embeddings
        self.num_embeddings_tp = divide_up(num_embeddings, self.tp_size)
        start_idx = self.num_embeddings_tp * tp_rank
        finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)
        self.vocab_range = (start_idx, finish_idx - start_idx)
        self.weight = torch.empty(self.num_embeddings_tp, embedding_dim)
        self._comm = DistributedCommunicator()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Custom kernel handles vocab_range masking
        from minisgl.kernel import indexing
        y = indexing(
            weights=self.weight,
            indices=x,
            vocab_range=self.vocab_range if self.tp_size > 1 else None,
        )
        return self._comm.all_reduce(y) if self.tp_size > 1 else y


class ParallelLMHead(VocabParallelEmbedding):
    """
    Language model head (logits projection).

    WHEN: lm_head in model (final layer)
    WHY: Project hidden states to vocab logits

    Supports weight tying with embed_tokens.
    For prefill, only computes logits for last token per sequence.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        tie_word_embeddings: bool = False,
        tied_embedding: VocabParallelEmbedding | None = None,
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.bias = torch.empty(self.num_embeddings_tp) if bias else None
        self.tied_embedding = tied_embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        bs = batch.size

        # For prefill, only compute logits for last position
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(bs)
            x = x[indices].contiguous()
            del indices

        module = self.tied_embedding or self
        logits = F.linear(x, module.weight, self.bias)

        if self.tp_size == 1:
            return logits

        # All-gather and reshape for full vocab
        input_shape = logits.shape
        output_tensor = self._comm.all_gather(logits)
        if bs == 1:
            return output_tensor.view(1, -1)[:, :self.num_embeddings]
        output_tensor = output_tensor.view((self.tp_size,) + input_shape)
        output_tensor = output_tensor.movedim(0, -1)
        output_tensor = output_tensor.reshape(input_shape[:1] + (self.tp_size * input_shape[1],))
        return output_tensor[:, :self.num_embeddings]


# ==============================================================================
# Normalization (layers/norm.py)
# ==============================================================================


class RMSNorm(BaseOP):
    """
    Root Mean Square Layer Normalization.

    WHEN: QK-norm in Qwen3 (applied to Q and K)
    """
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import rmsnorm
        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        """In-place version for efficiency."""
        self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
    """
    Fused RMSNorm with residual addition.

    WHEN: input_layernorm, post_attention_layernorm, final norm
    WHY: Fuses norm + residual add for efficiency
    """
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import fused_add_rmsnorm, rmsnorm
        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual


# ==============================================================================
# Activation (layers/activation.py)
# ==============================================================================


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU activation with gating (SwiGLU).

    WHEN: In GatedMLP after gate_up_proj

    Input: [gate, up] concatenated
    Output: SiLU(gate) * up
    """
    from flashinfer import silu_and_mul
    return silu_and_mul(x)


# ==============================================================================
# Rotary Position Embeddings (layers/rotary.py)
# ==============================================================================


class RotaryEmbedding(StateLessOP):
    """
    Rotary Position Embeddings (RoPE).

    WHEN: In AttentionLayer, applied to Q and K
    WHY: Position information via rotation in embedding space

    Precomputes cos/sin cache for efficiency.
    """
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        post_process: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        if post_process is not None:
            inv_freq = post_process(inv_freq)  # For Llama3 rope scaling

        # Precompute cos/sin cache
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1)

        from flashinfer import apply_rope_with_cos_sin_cache_inplace
        self.apply_rope_with_cos_sin_cache_inplace = apply_rope_with_cos_sin_cache_inplace

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key in-place."""
        self.apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self._cos_sin_cache,
        )
        return query, key


_ROPE_DEVICE: torch.device | None = None


def set_rope_device(device: torch.device):
    """Set device for RoPE cache (must be called before model creation)."""
    global _ROPE_DEVICE
    _ROPE_DEVICE = device


@lru_cache()
def get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,
) -> RotaryEmbedding:
    """Get cached RoPE instance."""
    rope_map = dict(rope_scaling) if rope_scaling is not None else None

    # Handle Llama3 rope scaling
    if rope_map and rope_map.get("rope_type") == "llama3":
        import math
        scaling_factor = rope_map["factor"]
        low_freq_factor = rope_map["low_freq_factor"]
        high_freq_factor = rope_map["high_freq_factor"]
        original_max_position = rope_map["original_max_position_embeddings"]

        def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
            wave_len = 2 * math.pi / inv_freq
            if low_freq_factor == high_freq_factor:
                return torch.where(
                    wave_len < original_max_position / high_freq_factor,
                    inv_freq,
                    inv_freq / scaling_factor,
                )
            delta = high_freq_factor - low_freq_factor
            smooth = (original_max_position / wave_len - low_freq_factor) / delta
            smooth = torch.clamp(smooth, 0, 1)
            factor = (1 - smooth) / scaling_factor + smooth
            return factor * inv_freq

        return RotaryEmbedding(head_dim, rotary_dim, max_position, base, post_process)

    return RotaryEmbedding(head_dim, rotary_dim, max_position, base)


# ==============================================================================
# Attention Layer (layers/attention.py)
# ==============================================================================


class AttentionLayer(StateLessOP):
    """
    Multi-head attention layer.

    WHEN: Called by RopeAttn in each decoder layer

    Steps:
      1. Split QKV (already projected)
      2. Apply QK-norm if present (Qwen3)
      3. Apply RoPE
      4. Call attention backend (FlashInfer/FA3)
      5. Backend stores K/V to cache
    """
    def __init__(
        self,
        layer_id: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_config: RotaryConfig,
        q_norm: RMSNorm | None = None,
        k_norm: RMSNorm | None = None,
    ):
        assert num_qo_heads % num_kv_heads == 0
        self.layer_id = layer_id
        self.head_dim = head_dim
        tp_size = get_tp_info().size
        self.num_qo_heads = divide_even(num_qo_heads, tp_size)
        self.num_kv_heads = divide_even(num_kv_heads, tp_size)
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=rotary_config.rotary_dim,
            max_position=rotary_config.max_position,
            base=rotary_config.base,
            rope_scaling=tuple(rotary_config.scaling.items()) if rotary_config.scaling else None,
        )
        self.q_norm = q_norm
        self.k_norm = k_norm

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        metadata = ctx.batch.attn_metadata

        # Split merged QKV
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)

        # Optional QK-norm (Qwen3)
        if self.q_norm is not None:
            self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        if self.k_norm is not None:
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))

        # Apply RoPE
        if self.rotary:
            q, k = self.rotary.forward(metadata.positions, q, k)

        q = q.view(-1, self.num_qo_heads, self.head_dim)

        # Dispatch to attention backend (also stores K/V to cache)
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)

        return o.view(-1, self.qo_attn_dim)


# ##############################################################################
# PART 10: KV CACHE
# ##############################################################################
#
# WHEN: Allocated at engine init, accessed during attention
# WHY: Store K/V values for autoregressive generation


# ==============================================================================
# KV Cache Types (kvcache/base.py)
# ==============================================================================

import enum


class KVCacheLayout(enum.Enum):
    """Memory layout for KV cache."""
    LayerFirst = enum.auto()  # [2, num_layers, num_pages, heads, head_dim]
    PageFirst = enum.auto()   # [2, num_pages, num_layers, heads, head_dim]


class KVCacheType(enum.Enum):
    """Type of KV cache."""
    MHA = enum.auto()  # Multi-Head Attention


class BaseKVCache(ABC):
    """Abstract base for KV caches."""
    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor: ...
    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor: ...
    @abstractmethod
    def store_kv(self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int) -> None: ...
    @property
    @abstractmethod
    def device(self) -> torch.device: ...
    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...
    @property
    @abstractmethod
    def num_layers(self) -> int: ...


@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    """Handle for prefix cache matching (returned by match_prefix)."""
    cached_len: int


class SizeInfo:
    """Cache size information."""
    def __init__(self, evictable_size: int, protected_size: int):
        self.evictable_size = evictable_size
        self.protected_size = protected_size

    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size


class BaseCacheManager(ABC):
    """Abstract base for cache management with prefix caching."""
    @abstractmethod
    def match_prefix(self, input_ids: torch.Tensor) -> Tuple[BaseCacheHandle, torch.Tensor]: ...
    @abstractmethod
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None: ...
    @abstractmethod
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> int: ...
    @abstractmethod
    def evict(self, size: int) -> torch.Tensor: ...
    @property
    @abstractmethod
    def size_info(self) -> SizeInfo: ...


# ==============================================================================
# MHA KV Cache (kvcache/mha_pool.py)
# ==============================================================================


class MHAKVCache(BaseKVCache):
    """
    Multi-Head Attention KV Cache.

    WHEN: Created by Engine.__init__()
    WHY: Store K/V tensors for all layers

    Memory layout: [2, num_layers, num_pages, num_heads, head_dim]
    where 2 is for K and V.

    Page-based storage enables efficient prefix caching.
    """
    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        dtype: torch.dtype,
        kv_layout: KVCacheLayout,
        device: torch.device,
    ):
        tp_info = get_tp_info()
        local_kv_heads = divide_even(num_kv_heads, tp_info.size)

        match kv_layout:
            case KVCacheLayout.PageFirst:
                kv_buffer = torch.empty(
                    (2, num_pages, num_layers, local_kv_heads, head_dim),
                    device=device, dtype=dtype,
                ).permute(0, 2, 1, 3, 4)
            case KVCacheLayout.LayerFirst:
                kv_buffer = torch.empty(
                    (2, num_layers, num_pages, local_kv_heads, head_dim),
                    device=device, dtype=dtype,
                )

        self._kv_buffer = kv_buffer.view(2, num_layers, num_pages, 1, local_kv_heads, head_dim)
        self._num_layers = num_layers
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        self._storage_shape = (num_pages, local_kv_heads, head_dim)

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[index]

    def store_kv(self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int) -> None:
        """Store K/V to cache at specified page indices."""
        from minisgl.kernel import store_cache
        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc, k=k, v=v,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers


def create_kvcache(
    num_layers: int,
    num_kv_heads: int,
    num_pages: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    cache_layout: KVCacheLayout = KVCacheLayout.LayerFirst,
    cache_type: KVCacheType = KVCacheType.MHA,
) -> BaseKVCache:
    """Factory function to create KV cache."""
    if cache_type == KVCacheType.MHA:
        return MHAKVCache(
            num_kv_heads=num_kv_heads,
            num_pages=num_pages,
            kv_layout=cache_layout,
            num_layers=num_layers,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
        )
    raise ValueError(f"Unsupported KVCacheType: {cache_type}")


# ==============================================================================
# Radix Tree Cache Manager (kvcache/radix_manager.py) - Summary
# ==============================================================================
#
# The RadixCacheManager implements prefix caching using a radix tree:
#
# class RadixTreeNode:
#     children: Dict[int, RadixTreeNode]  # Token ID -> child node
#     ref_count: int                       # Number of active references
#     timestamp: int                       # Last access time (for LRU)
#     _key: torch.Tensor                   # Token IDs for this node
#     _value: torch.Tensor                 # KV cache page indices
#
# Key operations:
#
# match_prefix(input_ids):
#     - Walk tree matching tokens
#     - Return handle with cached_len and page indices
#     - Used by PrefillManager to skip cached prefix
#
# insert_prefix(input_ids, indices):
#     - Add new prefix to tree after request completes
#     - Enables future requests with same prefix to reuse KV
#
# evict(size):
#     - LRU eviction of unreferenced prefixes
#     - Called when cache is full
#
# WHY: Prefix caching dramatically speeds up similar prompts
# (e.g., same system prompt, few-shot examples)


# ##############################################################################
# PART 11: ATTENTION BACKENDS
# ##############################################################################
#
# WHEN: Called by AttentionLayer.forward()
# WHY: Different implementations optimized for different hardware/scenarios


# ==============================================================================
# Base Attention Backend (attention/base.py)
# ==============================================================================


@dataclass
class BaseAttnMetadata(ABC):
    """
    Metadata for attention computation.

    Contains positions and backend-specific data needed for attention.
    """
    positions: torch.Tensor

    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor:
        """Get indices of last token for each sequence (for logit computation)."""
        ...


class BaseAttnBackend(ABC):
    """
    Abstract base for attention backends.

    Implementations:
      - FlashInfer (fi): Optimized for decode, supports various features
      - FlashAttention3 (fa3): Optimized for prefill on Hopper GPUs
    """
    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch
    ) -> torch.Tensor:
        """
        Compute attention and store K/V to cache.

        Args:
            q: Query tensor [num_tokens, num_heads, head_dim]
            k: Key tensor [num_tokens, num_kv_heads, head_dim] (pre-RoPE)
            v: Value tensor [num_tokens, num_kv_heads, head_dim]
            layer_id: Which layer (for cache storage)
            batch: Current batch with metadata

        Returns:
            Output tensor [num_tokens, num_heads, head_dim]
        """
        ...

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None:
        """Prepare attention metadata before forward pass."""
        ...

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        """Initialize for CUDA graph capture."""
        ...

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None:
        """Prepare for CUDA graph capture."""
        ...

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None:
        """Prepare for CUDA graph replay."""
        ...


class HybridBackend(BaseAttnBackend):
    """
    Hybrid backend using different backends for prefill vs decode.

    WHEN: Default on Hopper (FA3 for prefill, FlashInfer for decode)
    WHY: Each backend optimized for its scenario
    """
    def __init__(self, prefill_backend: BaseAttnBackend, decode_backend: BaseAttnBackend):
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(self, q, k, v, layer_id, batch):
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch)

    def prepare_metadata(self, batch):
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.prepare_metadata(batch)

    def init_capture_graph(self, max_seq_len, bs_list):
        self.decode_backend.init_capture_graph(max_seq_len, bs_list)

    def prepare_for_capture(self, batch):
        self.decode_backend.prepare_for_capture(batch)

    def prepare_for_replay(self, batch):
        self.decode_backend.prepare_for_replay(batch)


def create_attention_backend(
    config: ModelConfig,
    base_kvcache: BaseKVCache,
    backend: str,
    page_table: torch.Tensor,
) -> BaseAttnBackend:
    """
    Factory function to create attention backend.

    Options:
      - "fa3": FlashAttention3 (best for prefill on Hopper)
      - "fi": FlashInfer (best for decode, works everywhere)
      - "fa3,fi": Hybrid (FA3 prefill, FI decode)
      - "auto": Auto-select based on GPU
    """
    if backend == "auto":
        if is_sm100_supported():  # Blackwell
            backend = "fi"
        elif is_sm90_supported():  # Hopper
            backend = "fa3,fi"
        else:
            backend = "fi"

    if "," in backend:
        p_backend, d_backend = backend.split(",", 1)
        if p_backend != d_backend:
            p = create_attention_backend(config, base_kvcache, p_backend, page_table)
            d = create_attention_backend(config, base_kvcache, d_backend, page_table)
            return HybridBackend(p, d)
        backend = p_backend

    # Import actual implementations
    if backend == "fa3":
        from minisgl.attention.fa3 import FlashAttentionBackend
        return FlashAttentionBackend(config, base_kvcache, page_table)
    elif backend == "fi":
        from minisgl.attention.fi import FlashInferBackend
        return FlashInferBackend(config, base_kvcache, page_table)

    raise ValueError(f"Unsupported attention backend: {backend}")


# ==============================================================================
# FlashInfer Backend (attention/fi.py) - Summary
# ==============================================================================
#
# class FlashInferBackend(BaseAttnBackend):
#     """
#     FlashInfer-based attention backend.
#
#     Features:
#       - Paged attention with page_size=1
#       - CUDA graph support for decode
#       - GQA (Grouped Query Attention) support
#
#     Uses flashinfer library for efficient attention kernels.
#     """
#
# For prefill:
#   - Uses ragged attention (variable sequence lengths)
#   - Stores K/V to cache
#
# For decode:
#   - Uses paged attention
#   - Loads K/V from cache
#   - CUDA graph captured for efficiency


# ==============================================================================
# FlashAttention3 Backend (attention/fa3.py) - Summary
# ==============================================================================
#
# class FlashAttentionBackend(BaseAttnBackend):
#     """
#     FlashAttention3-based backend (Hopper optimized).
#
#     Best for:
#       - Prefill phase on Hopper GPUs
#       - Long sequences
#
#     Uses flash_attn library with FP8 support on Hopper.
#     """


# ##############################################################################
# PART 12: UTILITIES
# ##############################################################################


def divide_even(a: int, b: int) -> int:
    """Divide a by b, asserting exact divisibility."""
    assert a % b == 0, f"{a} must be divisible by {b}"
    return a // b


def divide_up(a: int, b: int) -> int:
    """Divide a by b, rounding up."""
    return (a + b - 1) // b


@lru_cache(maxsize=None)
def _get_torch_cuda_version() -> Tuple[int, int] | None:
    """Get CUDA compute capability."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability()


def is_arch_supported(major: int, minor: int = 0) -> bool:
    """Check if GPU supports given compute capability."""
    arch = _get_torch_cuda_version()
    return arch is not None and arch >= (major, minor)


def is_sm90_supported() -> bool:
    """Check for Hopper (SM90) support."""
    return is_arch_supported(9, 0)


def is_sm100_supported() -> bool:
    """Check for Blackwell (SM100) support."""
    return is_arch_supported(10, 0)


"""
================================================================================
END OF CONSOLIDATED CODEBASE
================================================================================

SUMMARY:
--------
This file presents minisgl in execution order for serving Qwen3:

1. STARTUP: Environment, config, server launch
2. CORE: SamplingParams, Req, Batch, Context - the data flowing through
3. DISTRIBUTED: TP info and communication (all_reduce, all_gather)
4. SCHEDULER: The brain - manages requests, batching, caching
5. ENGINE: The executor - runs model, samples tokens
6. MODELS: Qwen3/Llama architectures
7. LAYERS: Building blocks (linear, norm, attention, RoPE)
8. KV CACHE: Memory management with prefix caching
9. ATTENTION: FA3 and FlashInfer backends

KEY INSIGHTS:
-------------
- Prefix caching via radix tree dramatically speeds up similar prompts
- CUDA graphs capture decode for minimal kernel launch overhead
- Overlap scheduling hides CPU latency during GPU compute
- Tensor parallelism splits vocab (embedding), heads (attention), FFN across GPUs
- Page-based KV cache enables efficient memory management

================================================================================
"""
