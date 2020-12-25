import subprocess
import shlex
import torch

class Gpu_used():
    def __init__(self):
        self.total = 0
        self.used = 0
        self.used_max = 0
        self.used_list = []

    def gpuinfo(self):
        command = 'nvidia-smi -q -d MEMORY | sed -n "/FB Memory Usage/,/Free/p" | sed -e "1d" -e "4d" -e "s/ MiB//g" | cut -d ":" -f 2 | cut -c2-'
        # command = 'nvidia-smi -q -d MEMORY | sed -n "/BAR1 Memory Usage/,/Free/p" | sed -e "1d" -e "4d" -e "s/ MiB//g" | cut -d ":" -f 2 | cut -c2-'
        commands = [shlex.split(part) for part in command.split(' | ')]
        for i, cmd in enumerate(commands):
            if i==0:
                res = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            else:
                res = subprocess.Popen(cmd, stdin=res.stdout, stdout=subprocess.PIPE)
        self.total, self.used = map(int, res.communicate()[0].decode('utf-8').strip().split('\n'))

        info = {'total_MiB':self.total, 'used_MiB':self.used}
        if self.used > self.used_max:
          self.used_max = self.used
        self.used_list.append(self.used)
        return info
    
    def gpu_clear(self):
        torch.cuda.empty_cache()

    def gpu_property(self, device, torch_version=False):
        if torch_version:
            print(f'Using torch {torch.__version__}')
        return torch.cuda.get_device_properties(device)


# from verification.gpu import Gpu_used
# gpu_used = Gpu_used()
# print(gpu_used.gpuinfo())
# print(gpu_used.used_max)