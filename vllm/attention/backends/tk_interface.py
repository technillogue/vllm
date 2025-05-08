from typing import List, Tuple

import torch
from torch import Tensor
import thunderkittens as gqa_decode_4_heads as gqa_decode_4_heads_, create_schedule

# repeating this so we have keyword arguments instead of positional only
def gqa_decode_4_heads(
    instructions: Tensor,
    Q: Tensor,
    K_cache: Tensor,
    V_cache: Tensor,
    K_new: Tensor,
    V_new: Tensor,
    cos: Tensor,
    sin: Tensor,
    Table: Tensor,
    O: Tensor,
    O_scratch: Tensor,
    Lvec_scratch: Tensor,
    semaphore: Tensor,
    Softmax_scale: float,
    tic: int,
):
    return gqa_decode_4_heads_(
        instructions,
        Q,
        K_cache,
        V_cache,
        K_new,
        V_new,
        cos,
        sin,
        Table,
        O,
        O_scratch,
        Lvec_scratch,
        semaphore,
        Softmax_scale,
        tic,
    )


HIDDEN_DIM = 128
Q_HEADS = 8

# PAGE_SIZE = 256
# PAGE_BUFFER_SIZE = 10
# NUM_PAGES = 10000

# KV_CACHE_ROW_SIZE = 32

NPROC = 132


class Scheduler:
    create_schedule = staticmethod(create_schedule)

    @staticmethod
    def create_instructions(
        tasks: List,
    ) -> Tuple[torch.Tensor, int]:
        OP_PARTIAL, OP_REDUCTION = 1, 2

        num_instructions = max(t.uid for t in tasks) + 1
        processor_tasks = [[] for _ in range(NPROC)]

        for task in tasks:
            processor_tasks[task.processor].append(task)

        for pid in range(NPROC):
            processor_tasks[pid].sort(key=lambda t: t.start)

        max_num_processor_instructions = max(len(ptasks) for ptasks in processor_tasks)

        instructions = torch.zeros(
            (NPROC, max_num_processor_instructions, 32), dtype=torch.int32
        )

        for pid in range(NPROC):
            for tid, task in enumerate(processor_tasks[pid]):
                if task.task_type == OP_PARTIAL:
                    instructions[pid, tid, 0] = OP_PARTIAL
                    instructions[pid, tid, 1] = task.uid
                    instructions[pid, tid, 2] = (
                        -task.uid - 1 if task.write_scratch else task.batch_id
                    )
                    instructions[pid, tid, 3] = min(task.token_ids)
                    instructions[pid, tid, 4] = task.batch_id
                    instructions[pid, tid, 5] = min(task.token_ids)
                    instructions[pid, tid, 6] = task.start_seq
                    instructions[pid, tid, 7] = task.end_seq
                    instructions[pid, tid, 8] = task.seqlen

                elif task.task_type == OP_REDUCTION:
                    instructions[pid, tid, 0] = OP_REDUCTION
                    instructions[pid, tid, 1] = task.uid
                    instructions[pid, tid, 2] = len(task.dependencies) - 1
                    instructions[pid, tid, 3] = (
                        -task.uid - 1 if task.write_scratch else task.batch_id
                    )
                    instructions[pid, tid, 4] = task.token_ids[0]

                    for i, dep in enumerate(task.dependencies):
                        if i < 27:
                            instructions[pid, tid, 5 + i] = dep

                else:
                    raise ValueError(f"Unknown task type: {task.task_type}")

        Instructions = instructions.cuda()

        return Instructions, num_instructions


# def get_tensors(seqlens: List[int], decode_seqlen: int, return_tasks: bool = False):
#     Tasks = Scheduler.create_schedule(seqlens, decode_seqlen)

#     Instructions, num_instructions = Scheduler.create_instructions(Tasks)

#     max_num_processor_instructions = Instructions.shape[1]

#     O_scratch = torch.zeros(
#         (num_instructions, decode_seqlen, Q_HEADS, HIDDEN_DIM),
#         dtype=torch.float32,
#         device="cuda",
#     )

#     Lvec_scratch = torch.zeros(
#         (num_instructions, decode_seqlen, Q_HEADS),
#         dtype=torch.float32,
#         device="cuda",
#     )

#     Semaphore = torch.zeros(
#         (num_instructions, decode_seqlen), dtype=torch.int32, device="cuda"
#     )

#     Timings = torch.zeros(
#         (NPROC, max_num_processor_instructions, 64),
#         dtype=torch.int32,
#         device="cuda",
#     )

#     if return_tasks:
#         return Instructions, O_scratch, Lvec_scratch, Semaphore, Timings, Tasks
#     else:
#         return Instructions, O_scratch, Lvec_scratch, Semaphore, Timings, None

def gqa_decode_cuda(
    # The QKV components.
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    # The cached KV components.
    kcache: torch.Tensor, vcache: torch.Tensor,
    # The block table for paged-kv.
    block_table: torch.Tensor,
    # The cumulative query and key length specific(s) for varlen decode.
    cu_seqlen_query: torch.Tensor,
    # The prefix lengths for constraining the KV subset.
    prefix_seqlens: torch.Tensor,
    # The cached rotary embedding supports.
    cos: torch.Tensor, sin: torch.Tensor,
    # The softmax scale in attention.
    softmax_scale: float,
    # Non-sliding window attention.
    left_window: int, right_window: int,
    # Causal self-attention.
    causal: bool,
    # Non-interleaved rotary embedding.
    interleaved: bool,
    # Attention logits soft-capping.
    softcap: float,
    # Optional qkv scales when using fp8 attention.
    scales_q: torch.Tensor, scales_k: torch.Tensor, scales_v: torch.Tensor,
    rank: int,
    layer_id: int,

    # ThunderGQA specifics.
    # seqlens
    cache_seqlens: torch.Tensor,
    new_tokens: int = 1
    # num_instructions: int,
    # instructions: torch.Tensor,
) -> torch.Tensor:
    
    if cu_seqlen_query is not None:
        raise NotImplementedError("Varlen is not supported in ThunderGQA.")
    
    if not causal:
        raise NotImplementedError("Non-Causal attention is not supported in ThunderGQA.")
    
    if scales_q is not None or scales_k is not None or scales_v is not None:
        raise NotImplementedError("FP8 attention is not supported in ThunderGQA.")
    
    if interleaved:
        raise NotImplementedError("Interleaved rotary embedding is not supported in ThunderGQA.")
    
    if prefix_seqlens:
        raise NotImplementedError("Prefix sharing is not supported in ThunderGQA.")

    tasks = Scheduler.create_schedule(cache_seqlens, new_tokens, NPROCS)
    instructions, num_instructions = Scheduler.create_instructions(tasks)

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    _, decode_length, num_heads, head_size = query.size()
    device = query.device

    output_scratch = torch.zeros(
        (num_instructions, decode_length, num_heads, head_size),
        dtype=torch.float32,
        device=device,
    )

    intermediate_scratch = torch.zeros(
        (num_instructions, decode_length, num_heads),
        dtype=torch.float32,
        device=device,
    )

    semaphore = torch.zeros(
        (num_instructions, decode_length),
        dtype=torch.int32,
        device=device,
    )

    attn_out = torch.zeros_like(query)

    num_gpu_sm, processors = instructions.size(0), instructions.size(1)

    timings = (
        torch.zeros(
            (num_gpu_sm, processors, 64),
            dtype=torch.int32,
            device=instructions.device,
        )
    )

    gqa_decode_8_heads(
        instructions=instructions,
        Q=query,
        K_cache=kcache,
        V_cache=vcache,
        K_new=key,
        V_new=value,
        cos=cos,
        sin=sin,
        Table=block_table,
        O=attn_out,
        O_scratch=output_scratch,
        Lvec_scratch=intermediate_scratch,
        semaphore=semaphore,
        Softmax_scale=softmax_scale,
        tic=1, # tic
        instructions=timings,
    )

    return attn_out
