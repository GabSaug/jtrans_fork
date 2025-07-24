#!/usr/bin/env python3
import os
import subprocess
import multiprocessing
import time
from util.pairdata import pairdata
from os.path import basename
from pathlib import Path
from tqdm import tqdm

ida_path = f"{Path.home()}/idapro-7.5/idat64"
work_dir = os.path.abspath('.')
dataset_dir = "./dataset/"
script_path = "./process.py"
SAVE_ROOT = "./extract"

# make sure our dirs exist
for folder in [dataset_dir, SAVE_ROOT, "log", "idb"]:
    os.makedirs(folder, exist_ok=True)

def getTarget(path, prefixfilter=None):
    target = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if prefixfilter is None:
                target.append(os.path.join(root, file))
            else:
                for prefix in prefixfilter:
                    if file.startswith(prefix):
                        target.append(os.path.join(root, file))
    return target

def run_ida(cmd):
    """Invoke IDA and return its exit code."""
    return subprocess.call(cmd)

if __name__ == '__main__':
    start = time.time()

    target_list = getTarget(dataset_dir)

    tasks = []
    pool = multiprocessing.Pool(processes=1)

    # 1) Launching
    for target in target_list:
        filename = basename(target)
        cmd = [
            ida_path,
            f'-Llog/{filename}.log',
            '-c', '-A',
            f'-S{script_path}',
            f'-oidb/{filename}.idb',
            target
        ]
        print(cmd)
        ar = pool.apply_async(run_ida, args=(cmd,))
        tasks.append((filename, cmd, ar))

    pool.close()
    pool.join()

    # 2) Collecting results
    success_count = 0
    failure_count = 0
    failure_log_path = "log/failures.log"
    with open(failure_log_path, "w") as flog:
        for filename, cmd, ar in tqdm(tasks, desc="Collecting results"):
            rc = ar.get()
            if rc == 0:
                success_count += 1
            else:
                failure_count += 1
                flog.write(f"{filename}: return code {rc}\n")
                flog.write("    " + " ".join(cmd) + "\n\n")

    print(f"[*] Features Extracting Done")
    print(f"    Successes: {success_count}")
    print(f"    Failures:  {failure_count} (see {failure_log_path})")

    pairdata(SAVE_ROOT)

    end = time.time()
    print(f"[*] Time Cost: {end - start:.1f} seconds")

