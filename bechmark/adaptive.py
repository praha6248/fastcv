import time
import cv2
import torch
import fastcv
import numpy as np

def benchmark_adaptive(sizes=[1024, 2048, 4096], runs=50):
    results = []
    max_val = 255
    blockSize = 11
    C = 2
    method = 0 
    thresh_type = 0

    for size in sizes:
        print(f"\nBenchmarking {size}x{size} grayscale image")
        img_np = np.random.randint(0, 256, (size, size), dtype=np.uint8)
        img_torch = torch.from_numpy(img_np)

        start = time.perf_counter()
        for _ in range(runs):
            _ = cv2.adaptiveThreshold(img_np, max_val, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, blockSize, C)
        end = time.perf_counter()
        cv_time = (end - start) / runs * 1000

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            _ = fastcv.adaptive_threshold_tensor(img_torch, max_val, method, thresh_type, blockSize, C)
        torch.cuda.synchronize()
        end = time.perf_counter()
        fc_time = (end - start) / runs * 1000

        results.append((size, cv_time, fc_time))
        print(f"OpenCV (CPU): {cv_time:.4f} ms | fastcv (CUDA Wrapper): {fc_time:.4f} ms")
    
    return results

if __name__ == "__main__":
    results = benchmark_adaptive()
    print("\nFinal Results")
    print("Size\t\tOpenCV (CPU)\tfastcv (CUDA)")
    for size, cv_time, fc_time in results:
        print(f"{size}x{size}\t{cv_time:.4f} ms\t{fc_time:.4f} ms")