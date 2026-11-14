"""
scripts/profile_latency.py
Profiles inference turnaround time for IntentParser over 100 runs.
Supports both CPU and GPU (using CUDA synchronizations for exact timings).
"""
import time
import torch
import numpy as np
from src.intent_parser import IntentParser

BENCHMARK_PROMPTS = [
    "kinetic energy equals half mass times velocity squared",
    "the delete operation in red-black trees preserves balance through rotations",
    "replace momentum with force",
    "scratch that last word",
    "read back the active paragraph",
]

def run_latency_benchmark(model_dir="models/intent_slot_model", iterations=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=======================================================")
    print(f"  BENCHMARKING INTENT PARSER INFERENCE ON: {device.upper()}")
    print(f"=======================================================")
    
    parser = IntentParser(model_dir=model_dir)
    latencies = []

    # 1. Warm-up Phase (10 iterations to prime memory & torch kernels)
    print("Executing 10 warm-up iterations...")
    for _ in range(10):
        _ = parser.parse(BENCHMARK_PROMPTS[0])

    # 2. Benchmark Phase (100 timed iterations)
    print(f"Profiling {iterations} benchmark iterations...")
    for i in range(iterations):
        prompt = BENCHMARK_PROMPTS[i % len(BENCHMARK_PROMPTS)]
        
        if device == "cuda":
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = parser.parse(prompt)
            torch.cuda.synchronize()
            end = time.perf_counter()
        else:
            start = time.perf_counter()
            _ = parser.parse(prompt)
            end = time.perf_counter()

        latencies.append((end - start) * 1000.0)  # ms

    # 3. Statistical Calculations
    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    median_lat = np.median(latencies)
    p95_lat = np.percentile(latencies, 95)
    p99_lat = np.percentile(latencies, 99)

    print("\n--- INFERENCE LATENCY BENCHMARK RESULTS ---")
    print(f"Device:                 {device.upper()}")
    print(f"Sample Size:            {iterations} iterations")
    print(f"Mean Latency:           {mean_lat:.2f} ms ± {std_lat:.2f} ms")
    print(f"Median Latency:         {median_lat:.2f} ms")
    print(f"95th Percentile (P95):  {p95_lat:.2f} ms")
    print(f"99th Percentile (P99):  {p99_lat:.2f} ms")
    
    meets_target = p95_lat <= 50.0
    print(f"Target Threshold (<=50ms): {'PASSED [OK]' if meets_target else 'FAILED'}")
    
    return {
        "device": device.upper(),
        "mean": f"{mean_lat:.2f} ms",
        "std": f"{std_lat:.2f} ms",
        "median": f"{median_lat:.2f} ms",
        "p95": f"{p95_lat:.2f} ms"
    }

if __name__ == "__main__":
    _ = run_latency_benchmark()