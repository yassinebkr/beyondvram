"""Scheduling variants for the layer-streaming overlap experiment."""
from __future__ import annotations

from dataclasses import dataclass, field
import threading

import torch


@dataclass
class StageTimes:
    """Per-block service times, measured independently of end-to-end makespan."""

    read: list[float] = field(default_factory=list)
    stage_copy: list[float] = field(default_factory=list)
    h2d: list[float] = field(default_factory=list)
    compute: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "read": self.read,
            "stage_copy": self.stage_copy,
            "h2d": self.h2d,
            "compute": self.compute,
        }


@dataclass
class PipelineContext:
    """Owned buffers and operations shared by scheduling variants."""

    device: torch.device
    blocks: int
    block_bytes: int
    storage: object
    pinned: list
    device_buffers: list
    left: torch.Tensor
    right: torch.Tensor
    out: torch.Tensor
    sums: torch.Tensor
    copy_stream: torch.cuda.Stream

    def gemm(self) -> None:
        torch.mm(self.left, self.right, out=self.out)


def _timed_pair() -> tuple[torch.cuda.Event, torch.cuda.Event]:
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def run_sync(ctx: PipelineContext) -> StageTimes:
    """Variant 1: fully synchronous read -> stage -> H2D -> compute per block."""
    times = StageTimes()
    device_buffer = ctx.device_buffers[0]
    pinned = ctx.pinned[0]
    for block in range(ctx.blocks):
        read_s, copy_s = ctx.storage.read_next_into(pinned)
        times.read.append(read_s)
        times.stage_copy.append(copy_s)
        h2d_start, h2d_end = _timed_pair()
        h2d_start.record()
        device_buffer.copy_(pinned, non_blocking=False)
        h2d_end.record()
        compute_start, compute_end = _timed_pair()
        compute_start.record()
        ctx.gemm()
        ctx.sums[block] = device_buffer.sum(dtype=torch.int64)
        compute_end.record()
        torch.cuda.synchronize()
        times.h2d.append(h2d_start.elapsed_time(h2d_end) / 1000)
        times.compute.append(compute_start.elapsed_time(compute_end) / 1000)
    return times


def run_double_buffer(ctx: PipelineContext) -> StageTimes:
    """Variant 2: H2D(N+1) overlaps compute(N); storage remains synchronous.

    A pinned slot is not overwritten until its preceding H2D is complete, and a
    device slot is not refilled until the preceding compute has consumed it.
    """
    times = StageTimes()
    blocks = ctx.blocks
    compute_stream = torch.cuda.current_stream()
    ev_h2d = [torch.cuda.Event() for _ in range(blocks)]
    ev_compute = [torch.cuda.Event() for _ in range(blocks)]
    h2d_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    compute_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def enqueue_h2d(block: int) -> None:
        slot = block % 2
        pair = _timed_pair()
        with torch.cuda.stream(ctx.copy_stream):
            pair[0].record()
            ctx.device_buffers[slot].copy_(ctx.pinned[slot], non_blocking=True)
            pair[1].record()
        h2d_pairs.append(pair)
        ev_h2d[block].record(ctx.copy_stream)

    read_s, copy_s = ctx.storage.read_next_into(ctx.pinned[0])
    times.read.append(read_s)
    times.stage_copy.append(copy_s)
    enqueue_h2d(0)

    for block in range(blocks):
        if block + 1 < blocks:
            next_slot = (block + 1) % 2
            if block > 0:
                ev_h2d[block - 1].synchronize()
                ctx.copy_stream.wait_event(ev_compute[block - 1])
            read_s, copy_s = ctx.storage.read_next_into(ctx.pinned[next_slot])
            times.read.append(read_s)
            times.stage_copy.append(copy_s)
            enqueue_h2d(block + 1)
        compute_stream.wait_event(ev_h2d[block])
        pair = _timed_pair()
        pair[0].record(compute_stream)
        ctx.gemm()
        ctx.sums[block] = ctx.device_buffers[block % 2].sum(dtype=torch.int64)
        pair[1].record(compute_stream)
        compute_pairs.append(pair)
        ev_compute[block].record(compute_stream)

    torch.cuda.synchronize()
    times.h2d = [start.elapsed_time(end) / 1000 for start, end in h2d_pairs]
    times.compute = [start.elapsed_time(end) / 1000 for start, end in compute_pairs]
    return times


def run_three_stage(ctx: PipelineContext) -> StageTimes:
    """Variant 3: worker storage(N+2), H2D(N+1), and compute(N).

    The storage worker waits for the last H2D that used a three-slot pinned
    buffer. The copy stream waits for the compute that used each two-slot
    device buffer before refilling it.
    """
    times = StageTimes(read=[0.0] * ctx.blocks, stage_copy=[0.0] * ctx.blocks)
    blocks = ctx.blocks
    compute_stream = torch.cuda.current_stream()
    storage_done = [threading.Event() for _ in range(blocks)]
    h2d_enqueued = [threading.Event() for _ in range(blocks)]
    ev_h2d = [torch.cuda.Event() for _ in range(blocks)]
    ev_compute = [torch.cuda.Event() for _ in range(blocks)]
    h2d_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    compute_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    worker_errors: list[BaseException] = []

    def storage_worker() -> None:
        try:
            for block in range(blocks):
                slot = block % 3
                if block >= 3:
                    h2d_enqueued[block - 3].wait()
                    ev_h2d[block - 3].synchronize()
                read_s, copy_s = ctx.storage.read_next_into(ctx.pinned[slot])
                times.read[block] = read_s
                times.stage_copy[block] = copy_s
                storage_done[block].set()
        except BaseException as exc:
            worker_errors.append(exc)
            for event in storage_done:
                event.set()

    def wait_for_storage(block: int) -> None:
        storage_done[block].wait()
        if worker_errors:
            raise RuntimeError("storage worker failed") from worker_errors[0]

    def enqueue_h2d(block: int) -> None:
        pair = _timed_pair()
        with torch.cuda.stream(ctx.copy_stream):
            if block >= 2:
                ctx.copy_stream.wait_event(ev_compute[block - 2])
            pair[0].record()
            ctx.device_buffers[block % 2].copy_(ctx.pinned[block % 3], non_blocking=True)
            pair[1].record()
        h2d_pairs.append(pair)
        ev_h2d[block].record(ctx.copy_stream)
        h2d_enqueued[block].set()

    worker = threading.Thread(target=storage_worker, name="overlap-storage-stage")
    worker.start()
    try:
        wait_for_storage(0)
        enqueue_h2d(0)
        for block in range(blocks):
            if block + 1 < blocks:
                wait_for_storage(block + 1)
                enqueue_h2d(block + 1)
            compute_stream.wait_event(ev_h2d[block])
            pair = _timed_pair()
            pair[0].record(compute_stream)
            ctx.gemm()
            ctx.sums[block] = ctx.device_buffers[block % 2].sum(dtype=torch.int64)
            pair[1].record(compute_stream)
            compute_pairs.append(pair)
            ev_compute[block].record(compute_stream)
    finally:
        worker.join()

    torch.cuda.synchronize()
    times.h2d = [start.elapsed_time(end) / 1000 for start, end in h2d_pairs]
    times.compute = [start.elapsed_time(end) / 1000 for start, end in compute_pairs]
    return times
