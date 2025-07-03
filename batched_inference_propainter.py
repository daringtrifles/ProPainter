import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
mask_dilation = 5
def gpu_worker(gpu_id: int, job_queue: Queue):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    while True:
        item = job_queue.get()
        if item is None:
            break
        video_dir, mask_dir, out_dir = item
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "inference_propainter.py",
            "--video", str(video_dir),
            "--mask", str(mask_dir),
            "--output", str(out_dir),
            "--fp16",
            "--subvideo_length", "10",
            "--mask_dilation", f"{mask_dilation}",
        
        ]
        print(f"[GPU {gpu_id}] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=env)
        job_queue.task_done()

# -----------------------------------
# Main
# -----------------------------------
def main():
    parser = argparse.ArgumentParser()
    #mask frames are expected to be in the form of mask_root/episode_number/mask_frames/
    #video frames are expected to be in the form of video_root/episode_number/frames/
    parser.add_argument("--directory_root", required=True, type=Path)
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--gpus", required=True, nargs="+", type=int)
    parser.add_argument("--results_root", default=Path("results"), type=Path)
    parser.add_argument("--mask_dilation", default=5, type=int)
    args = parser.parse_args()
    mask_dilation = args.mask_dilation

# Find all episode_* directories
    jobs = []
    for i in range(args.start, args.end + 1):
        video_dir = args.directory_root / f"{i}" / "frames"
        mask_dir = args.directory_root / f"{i}" / "mask_frames"
        out_dir = args.results_root / f"{i}"
        jobs.append((video_dir, mask_dir, out_dir))
    print(f"Found {len(jobs)} jobs")

    job_q = Queue()
    for job in jobs:
        job_q.put(job)
    for _ in args.gpus:
        job_q.put(None)  # poison pill

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        for gpu in args.gpus:
            executor.submit(gpu_worker, gpu, job_q)

    job_q.join()
    print("✅ All jobs finished.")

if __name__ == "__main__":
    main()