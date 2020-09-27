#データセット
import torch.nn as nn
import torch.tensor
import pytorch.config
from efficientnet_pytorch import EfficientNet
print("クラス一覧")
print("Ef_Net","Torch_Dataset")

#コンフィグの読み直し！！
from pytorch import config
import pytorch
import importlib
importlib.reload(pytorch.config)
b_num = config.b_num

class Ef_Net(nn.Module):
    def __init__(self):
        super().__init__()
        #一度インスタンス化するために、archは変数に入れる必要性？？
        arch = EfficientNet.from_pretrained('efficientnet-'+b_num)
        self.arch = arch
        #in_features = {"b0":1000,"b1":1280,"b2":1408,"b3":1536,"b4":1792,"b5":2048,"b6":2304,"b7":2560}
        num_ftrs = self.arch._fc.in_features
        self.arch._fc = nn.Linear(in_features=num_ftrs, out_features=1)
        
    def print_attribute(self,key=True,value=True):
      for i, j in self.__dict__.items():
        if key==True and value == True:
          print("#########key#########",i, "#########value#########", j)
        elif value==False: 
          print(i)
        else:
          print(j)



    # def model_summary(self):
    #     # pip install torchsummary
    #     from torchsummary import summary
    #     summary(self.arch,(3,256,256)) # summary(model,(channel