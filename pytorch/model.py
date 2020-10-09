#モデル
# pip install torchsummary
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchsummary import summary

#自作モジュール
from pytorch import config
b_num = config.b_num
out_features = config.out_features
device = config.device
criterion = config.criterion

#Swishモジュール
sigmoid = torch.nn.Sigmoid()
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
swish = Swish.apply
class Swish_module(nn.Module):
    def forward(self, x):
        return swish(x)
# swish_layer = Swish_module()


class Ef_Net(nn.Module):
    def __init__(self, n_meta_features=0,out_features=out_features):
        super().__init__()
        #一度インスタンス化するために、archは変数に入れる必要性？？
        arch = EfficientNet.from_pretrained('efficientnet-'+b_num)
        self.arch = arch
        # Unfreeze model weights→凍結せず
        for param in self.arch.parameters():
            param.requires_grad = True

        #fc層の定義#in_features = {"b0":1000,"b1":1280,"b2":1408,"b3":1536,"b4":1792,"b5":2048,"b6":2304,"b7":2560}
        num_ftrs = self.arch._fc.in_features
        #meta
        # self.dropout = nn.Dropout(0.5)
        # if n_meta_features > 0:
        #     self.meta = nn.Sequential(
        #         nn.Linear(n_meta_features, 512),
        #         nn.BatchNorm1d(512),
        #         Swish_module(),
        #         nn.Dropout(p=0.3),
        #         nn.Linear(512, 128),
        #         nn.BatchNorm1d(128),
        #         Swish_module(),
        #     )
        #     num_ftrs += 128
        #転送する
        # self.myfc  = nn.Linear(in_features=num_ftrs, out_features=1)
        self.arch._fc  = nn.Linear(in_features=num_ftrs, out_features=out_features)
        # self.arch._fc = nn.Identity()
    
    #self.archで一階層深いところにネットワークを定義しているので、forwardが必須
    if str(criterion) == "BCELoss()":
        #BCELossの場合
        def forward(self, input):
          output = self.arch(input)
          m = nn.Sigmoid()
          output = m(output)
          return output
    else:
        def forward(self, input):
          #BCElogitの場合
          output = self.arch(input)
          return output
    # def extract(self, x):
    # x = self.arch(x)
    # return x


    #インスタンスのクラス属性の表示
    def print_attribute(self,key=True,value=True):
      for i, j in self.__dict__.items():
        if key==True and value == True:
          print("#########key#########",i, "#########value#########", j)
        elif value==False: 
          print(i)
        else:
          print(j)

    #モデルサマリーgpuセットしないとエラー吐く??
    def model_summary(self):
        self = self.to(device)
        summary(self.arch,(3,256,256))