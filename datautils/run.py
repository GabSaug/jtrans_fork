import os
import subprocess
import multiprocessing
import time
from util.pairdata import pairdata
from os.path import basename
from pathlib import Path

ida_path = "{}/idapro-7.5/idat64".format(Path.home())
work_dir = os.path.abspath('.')
dataset_dir = '../../../Binaries/Dataset-Muaz/'
script_path = "./process.py"
SAVE_ROOT = "./extract"

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


if __name__ == '__main__':
    # prefixfilter = ['libcap-git-setcap']
    start = time.time()
    target_list = getTarget(dataset_dir)

    pool = multiprocessing.Pool(processes=30)
    for target in target_list:
        filename = basename(target)
        #filename_strip = filename + '.strip'
        #ida_input = os.path.join(strip_path, filename_strip)
        ida_input = target
        #os.system(f"strip -s {target} -o {ida_input}")
        #print(f"strip -s {target} -o {ida_input}")

        cmd = [ida_path, f'-Llog/{filename}.log', '-c', '-A', f'-S{script_path}', f'-oidb/{filename}.idb', f'{ida_input}']
        print(cmd)
        pool.apply_async(subprocess.call, args=(cmd,))

    pool.close()
    pool.join()
    print('[*] Features Extracting Done')
    pairdata(SAVE_ROOT)
    end = time.time()
    print(f"[*] Time Cost: {end - start} seconds")
