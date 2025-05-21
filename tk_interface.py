from typing import List, Tuple
import os
import torch
from torch import Tensor
from thunderkittens import gqa_decode_4_heads as _gqa_decode_4_heads, create_schedule as _create_schedule
from torch import distributed as dist

#torch.set_printoptions(threshold=0, edgeitems=0, precision=1)

def create_rope_embeddings(seq_lengths, new_tokens, rope_dim, base: float = 10000.0):
    # add NUM_ROWS_d2 to account for loading in 16 rows to not overflow
    NUM_ROWS_d2 = 16
    seq_len = max(seq_lengths) + new_tokens + NUM_ROWS_d2

    t = torch.arange(seq_len, device=torch.device("cuda"), dtype=torch.float32)
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(
                0, rope_dim, 2, device=torch.device("cuda"), dtype=torch.float32
            )
            / rope_dim
        )
    )

    freqs = torch.outer(t, inv_freq)

    cos = torch.cos(freqs).to(torch.bfloat16)
    sin = torch.sin(freqs).to(torch.bfloat16)

    return cos, sin


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
    timings: Tensor,
):

    return _gqa_decode_4_heads(
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
        timings,
    )


HIDDEN_DIM = 128
Q_HEADS = 8


NPROC = 132


class Scheduler:
    create_schedule = staticmethod(_create_schedule)

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
                        -task.uid - 1 if task.args.write_scratch else task.batch_id
                    )
                    instructions[pid, tid, 3] = min(task.tok_ids)
                    instructions[pid, tid, 4] = task.batch_id
                    instructions[pid, tid, 5] = min(task.tok_ids)
                    instructions[pid, tid, 6] = task.args.start
                    instructions[pid, tid, 7] = task.args.end
                    instructions[pid, tid, 8] = task.args.length

                elif task.task_type == OP_REDUCTION:
                    instructions[pid, tid, 0] = OP_REDUCTION
                    instructions[pid, tid, 1] = task.uid
                    instructions[pid, tid, 2] = len(task.dependencies) - 1
                    instructions[pid, tid, 3] = (
                        -task.uid - 1 if task.args.write_scratch else task.batch_id
                    )
                    instructions[pid, tid, 4] = task.tok_ids[0]

                    for i, dep in enumerate(task.dependencies):
                        if i < 27:
                            instructions[pid, tid, 5 + i] = dep

                else:
                    raise ValueError(f"Unknown task type: {task.task_type}")

        Instructions = instructions.cuda()

        return Instructions, num_instructions

def task_repr(t):
    return (
        f"<Task uid={t.uid} batch_id={t.batch_id} tok_ids={t.tok_ids} name={t.name} "
        f"task_type={t.task_type} dependencies={t.dependencies} next_input_time={t.next_input_time} "
        f"start={t.start} finish={t.finish} processor={t.processor} "
        f"args=<Args start={t.args.start} end={t.args.end} length={t.args.length}>>"
    )


def get_scheduler_metadata(
    cache_seqlens,
    # I think this could be used for new_tokens
    cu_seqlens_q=None,
    # fa3 schedule has a lot of arguments we don't care about
    **kwargs
):
    # print("scheduler metadata", cache_seqlens, cu_seqlens_q)
    tasks = _create_schedule(seq_lengths=cache_seqlens.tolist(), new_tokens=1, num_processors=NPROC)
    instructions, _ = Scheduler.create_instructions(tasks)
    if dist.get_rank() == 0:
        print("tasks:", [task_repr(t) for t in tasks])
        print("instructions", instructions)
    #print("rank", os.environ.get("LOCAL_RANK"))
    #print("instructions", instructions)
    return instructions


def gqa_decode_cuda(
    # The QKV components.
    # query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    # The cached KV components.
    # kcache: torch.Tensor, vcache: torch.Tensor,
    k_cache: torch.Tensor, v_cache: torch.Tensor,
    # The block table for paged-kv.
    block_table: torch.Tensor,
    # The softmax scale in attention.
    softmax_scale: float,
    # Attention logits soft-capping.
    softcap: float,
    # cache_seqlens: torch.Tensor,
    seqused_k: torch.Tensor,
    # instructions tensor
    scheduler_metadata: torch.Tensor,

    # passed by vllm but not used:
    # max_seqlen_q, max_seqlen_k
    # window_size, alibi_slopes, fa_version
    # {q,k,v}_descale

    # # The cached rotary embedding supports.
    # # not provided by vllm?
    # cos: torch.Tensor, sin: torch.Tensor,

    # other engine features not supported
    # The cumulative query and key length specific(s) for varlen decode.
    # cu_seqlen_query: torch.Tensor,
    # The prefix lengths for constraining the KV subset.
    # prefix_seqlens: torch.Tensor,
    # Non-sliding window attention.
    # left_window: int, right_window: int,
    # Causal self-attention.
    causal: bool,
    # # Optional qkv scales when using fp8 attention.
    # scales_q: torch.Tensor = None, scales_k: torch.Tensor = None, scales_v: torch.Tensor = None,

    # unused
    **kwargs,
) -> torch.Tensor:
    # if cu_seqlen_query is not None:
    #     raise NotImplementedError("Varlen is not supported in ThunderGQA.")
    if not causal:
        raise NotImplementedError("Non-Causal attention is not supported in ThunderGQA.")
    # if scales_q is not None or scales_k is not None or scales_v is not None:
    #     raise NotImplementedError("FP8 attention is not supported in ThunderGQA.")
    # if interleaved:
    #     raise NotImplementedError("Interleaved rotary embedding is not supported in ThunderGQA.")
    # if prefix_seqlens:
    #     raise NotImplementedError("Prefix sharing is not supported in ThunderGQA.")

    # instructions have shape (num_processors, max_instructions_per_processor, 32)
    # dim 2 index 1 is uid
    num_instructions = 300 # scheduler_metadata[:, :, 1].max() + 1

    query = q.contiguous()

    *_, decode_length, num_heads, head_size = query.size()
    device = query.device

    # print("*_, decode_length, num_heads, head_size = query.size() =", query.size())
    # print("making output_scratch (num_instructions, decode_length, num_heads, head_size)", (num_instructions, decode_length, num_heads, head_size)) 
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

    num_gpu_sm, processors = scheduler_metadata.size(0), scheduler_metadata.size(1)

    cos, sin = create_rope_embeddings(seqused_k, 1, 64)

    timings = (
        torch.zeros(
            (num_gpu_sm, processors, 64),
            dtype=torch.int32,
            device=scheduler_metadata.device,
        )
    )
    # print("k/k_cache:", k_cache.shape, "block table:", block_table.shape, "instructions:", scheduler_metadata.shape)

    # have: [num_blocks, page_size, kv_heads, head_dim]
    # want [kv_heads=1, num_blocks, page_size, head_dim] 
    k_cache = k_cache.permute((2, 0, 1, 3)).contiguous()
    v_cache = v_cache.permute((2, 0, 1, 3)).contiguous()
    
    gqa_decode_4_heads(
        instructions=scheduler_metadata,
        Q=query,
        K_cache=k_cache,
        V_cache=v_cache,
        # K_new=k,
        # V_new=v,
        K_new=None,
        V_new=None,
        cos=cos,
        sin=sin,
        Table=block_table,
        O=attn_out,
        O_scratch=output_scratch,
        Lvec_scratch=intermediate_scratch,
        semaphore=semaphore,
        Softmax_scale=softmax_scale,
        tic=1, # tic
        timings=timings,
    )

    return attn_out
