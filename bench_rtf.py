#!/usr/bin/env python3
"""
Benchmark harness for SoulX-FlashTalk.

Measures latency, throughput, GPU utilization, VRAM peak, and computes
Real-Time Factor (RTF = processing_time / media_duration).

Usage:
    python bench_rtf.py \
        --ckpt_dir models/SoulX-FlashTalk-14B \
        --wav2vec_dir models/chinese-wav2vec2-base \
        --audio_path examples/cantonese_16k.wav \
        --cond_image examples/man.png \
        [--quantize quanto-int4] \
        [--cpu_offload] \
        [--num_chunks 5] \
        [--warmup_chunks 1]
"""
import argparse
import os
import sys
import time

import librosa
import numpy as np
import torch
from loguru import logger

from flash_talk.inference import (
    get_audio_embedding,
    get_base_data,
    get_pipeline,
    infer_params,
    run_pipeline,
)


def parse_args():
    parser = argparse.ArgumentParser(description="SoulX-FlashTalk RTF Benchmark")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Path to FlashTalk model checkpoint directory.")
    parser.add_argument("--wav2vec_dir", type=str, required=True,
                        help="Path to wav2vec checkpoint directory.")
    parser.add_argument("--audio_path", type=str, default="examples/cantonese_16k.wav",
                        help="Audio file for benchmarking.")
    parser.add_argument("--cond_image", type=str, default="examples/man.png",
                        help="Condition image.")
    parser.add_argument("--input_prompt", type=str,
                        default="A person is talking. Only the foreground characters are moving, the background remains static.",
                        help="Text prompt.")
    parser.add_argument("--base_seed", type=int, default=9999)
    parser.add_argument("--cpu_offload", action="store_true",
                        help="Enable CPU offload.")
    parser.add_argument("--quantize", type=str, default=None,
                        choices=["quanto-int4", "quanto-int8", "bitsandbytes-nf4"],
                        help="Quantization mode.")
    parser.add_argument("--num_chunks", type=int, default=5,
                        help="Number of video chunks to generate for the benchmark.")
    parser.add_argument("--warmup_chunks", type=int, default=1,
                        help="Number of warmup chunks (excluded from timing).")
    parser.add_argument("--audio_encode_mode", type=str, default="stream",
                        choices=["stream", "once"],
                        help="Audio encoding mode.")
    return parser.parse_args()


def get_gpu_memory_mb():
    """Return current and peak GPU memory in MB."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    current = torch.cuda.memory_allocated() / (1024 ** 2)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return current, peak


def main():
    args = parse_args()

    sample_rate = infer_params["sample_rate"]
    tgt_fps = infer_params["tgt_fps"]
    cached_audio_duration = infer_params["cached_audio_duration"]
    frame_num = infer_params["frame_num"]
    motion_frames_num = infer_params["motion_frames_num"]
    slice_len = frame_num - motion_frames_num

    # Compute media duration per chunk
    chunk_duration_sec = slice_len / tgt_fps  # seconds of video per chunk

    # ── Load pipeline ──────────────────────────────────────────────────
    logger.info("Loading pipeline...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    load_start = time.time()
    pipeline = get_pipeline(
        world_size=1,
        ckpt_dir=args.ckpt_dir,
        wav2vec_dir=args.wav2vec_dir,
        cpu_offload=args.cpu_offload,
        quantize_mode=args.quantize,
    )
    load_time = time.time() - load_start
    _, vram_after_load = get_gpu_memory_mb()
    logger.info(f"Pipeline loaded in {load_time:.1f}s  |  VRAM after load: {vram_after_load:.0f} MB")

    # ── Prepare base data ──────────────────────────────────────────────
    get_base_data(pipeline, input_prompt=args.input_prompt, cond_image=args.cond_image, base_seed=args.base_seed)

    # ── Load audio ─────────────────────────────────────────────────────
    human_speech_array_all, _ = librosa.load(args.audio_path, sr=sample_rate, mono=True)
    total_audio_duration = len(human_speech_array_all) / sample_rate
    logger.info(f"Audio duration: {total_audio_duration:.1f}s")

    # ── Prepare chunks ─────────────────────────────────────────────────
    from collections import deque

    cached_audio_length_sum = sample_rate * cached_audio_duration
    audio_end_idx = cached_audio_duration * tgt_fps
    audio_start_idx = audio_end_idx - frame_num

    audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)
    human_speech_array_slice_len = slice_len * sample_rate // tgt_fps
    human_speech_array_slices = human_speech_array_all[
        : (len(human_speech_array_all) // human_speech_array_slice_len) * human_speech_array_slice_len
    ].reshape(-1, human_speech_array_slice_len)

    available_chunks = len(human_speech_array_slices)
    total_needed = args.warmup_chunks + args.num_chunks
    if available_chunks < total_needed:
        logger.warning(
            f"Audio provides {available_chunks} chunks but {total_needed} requested "
            f"(warmup={args.warmup_chunks} + bench={args.num_chunks}). "
            f"Adjusting to {available_chunks} total."
        )
        total_needed = available_chunks
        args.num_chunks = max(1, total_needed - args.warmup_chunks)

    # ── Run benchmark ──────────────────────────────────────────────────
    logger.info(f"Running {args.warmup_chunks} warmup + {args.num_chunks} benchmark chunks...")

    chunk_times = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for chunk_idx in range(total_needed):
        human_speech_array = human_speech_array_slices[chunk_idx]

        # Audio encode
        audio_dq.extend(human_speech_array.tolist())
        audio_array = np.array(audio_dq)
        audio_embedding = get_audio_embedding(pipeline, audio_array, audio_start_idx, audio_end_idx)

        # Synchronize for accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()

        # Generate video chunk
        video = run_pipeline(pipeline, audio_embedding)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()

        elapsed = t1 - t0
        is_warmup = chunk_idx < args.warmup_chunks

        if not is_warmup:
            chunk_times.append(elapsed)

        tag = "WARMUP" if is_warmup else "BENCH "
        logger.info(f"  [{tag}] chunk {chunk_idx}: {elapsed:.3f}s  (media={chunk_duration_sec:.3f}s)")

        del video  # free memory

    # ── Report ─────────────────────────────────────────────────────────
    _, vram_peak = get_gpu_memory_mb()

    if not chunk_times:
        logger.error("No benchmark chunks were run.")
        sys.exit(1)

    chunk_times_arr = np.array(chunk_times)
    mean_time = chunk_times_arr.mean()
    std_time = chunk_times_arr.std()
    min_time = chunk_times_arr.min()
    max_time = chunk_times_arr.max()

    rtf = mean_time / chunk_duration_sec
    throughput_fps = slice_len / mean_time  # generated frames per second

    print("\n" + "=" * 64)
    print("  SoulX-FlashTalk Benchmark Results")
    print("=" * 64)
    print(f"  Quantization:       {args.quantize or 'None (bf16)'}")
    print(f"  CPU offload:        {args.cpu_offload}")
    print(f"  Chunks benchmarked: {args.num_chunks}")
    print(f"  Warmup chunks:      {args.warmup_chunks}")
    print(f"  Media per chunk:    {chunk_duration_sec:.3f}s ({slice_len} frames @ {tgt_fps} fps)")
    print("-" * 64)
    print(f"  Latency mean:       {mean_time:.3f}s ± {std_time:.3f}s")
    print(f"  Latency min/max:    {min_time:.3f}s / {max_time:.3f}s")
    print(f"  Throughput:         {throughput_fps:.1f} frames/s")
    print(f"  VRAM peak:          {vram_peak:.0f} MB")
    print(f"  Pipeline load time: {load_time:.1f}s")
    print("-" * 64)
    print(f"  *** RTF = {rtf:.3f} ***  (target ≤ 1.0)")
    if rtf <= 1.0:
        print("  ✅ PASS: Real-time factor target met!")
    else:
        print(f"  ❌ FAIL: RTF {rtf:.3f} > 1.0")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
