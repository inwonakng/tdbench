"""
Measures GPU memory every N second.
Returns the memory per gpu device in MBs.
source: https://stackoverflow.com/questions/67707828/how-to-get-every-seconds-gpu-usage-in-python
"""
import subprocess as sp
import torch
import pandas as pd
import time

def get_reserved_gpu_memory() -> list[int]:
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    try:
        memory_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    memory_use_values = [int(x.split()[0]) for _, x in enumerate(memory_use_info)]
    return memory_use_values

def get_active_gpu_memory() -> list[int]:
    return [
        torch.cuda.memory_allocated(device_num) // (1024**2)
        for device_num in range(torch.cuda.device_count())
    ]

def get_gpu_table() -> str:
    acts = get_active_gpu_memory()
    ress = get_reserved_gpu_memory()

    table = []
    for a,r in zip(acts, ress):
        table += [{
            "active": f"{a:,}MB",
            "reserved": f"{r:,}MB",
        }]

    table = pd.DataFrame(table)

    return table.to_markdown(index=False)

if __name__ == "__main__":
    while True:
        print(get_gpu_table())
        print("="*40)
        time.sleep(2)
