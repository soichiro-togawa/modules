#インポート
# pip install torchsummary
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchsummary import summary
#自作モジュール
from pytorch import config
#コンフィグ
b_num = config.MODEL_CONFIG["b_num"]
out_features = config.MODEL_CONFIG["out_features"]
device = config.MODEL_CONFIG["device"]
criterion = config.MODEL_CONFIG["criterion"]
use_meta = config.MODEL_CONFIG["use_meta"]

#Swishモジュール
sigmoid = nn.Sigmoid()
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
#applyはtorch.autograd.Functionの組み込み
swish = Swish.apply
class Swish_module(nn.Module):
    def forward(self, x):
        return swish(x)
# swish_layer = Swish_module()


class Ef_Net(nn.Module):
    def __init__(self, n_meta_features=0, out_features=out_features):
        super().__init__()
        #一度インスタンス化するために、archは変数に入れる必要性？？
        arch = EfficientNet.from_pretrained('efficientnet-'+b_num)
        self.arch = arch
        self.n_meta_features = n_meta_features
        self.dropout = nn.Dropout(0.5)
        # self.dropout_fold = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])
        # Unfreeze model weights→凍結せず
        for param in self.arch.parameters():
            param.requires_grad = True
        #fc層の定義#in_features = {"b0":1000,"b1":1280,"b2":1408,"b3":1536,"b4":1792,"b5":2048,"b6":2304,"b7":2560}
        num_ftrs = self.arch._fc.in_features

        #meta
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, 512),
                nn.BatchNorm1d(512),
                Swish_module(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                Swish_module(),
            )
            num_ftrs += 128
        #全結合層
        self.myfc  = nn.Linear(in_features=num_ftrs, out_features=out_features)
        #転送する(ほぼ削除と同義)
        self.arch._fc = nn.Identity()
        # self.arch._fc  = nn.Linear(in_features=num_ftrs, out_features=out_features)

    #criterionは外で定義してるのでifで使える
    if str(criterion) == "BCELoss()":
      def extract(self, x):
        sigmoid = nn.Sigmoid()
        x = self.arch(x)
        x = sigmoid(x)
        return x
    else:
      def extract(self, x):
        x = self.arch(x)
        return x

    #self.archで一階層深いところにネットワークを定義しているので、forwardが必須
    if use_meta:
        def forward(self, x, x_meta=None):
            x = self.extract(x).squeeze(-1).squeeze(-1)
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
            x = self.myfc(self.dropout(x))
            return x
    else:
        def forward(self, x, x_meta=None):
            x = self.myfc(self.dropout(x))
            return x

    #fc層のドロップアウトだけ複数回行う
    # def forward(self, x, x_meta=None):
    #     x = self.extract(x).squeeze(-1).squeeze(-1)
    #     if self.n_meta_features > 0:
    #         x_meta = self.meta(x_meta)
    #         x = torch.cat((x, x_meta), dim=1)
    #     for i, dropout in enumerate(self.dropout_fold):
    #         if i == 0:
    #             out = self.myfc(dropout(x))
    #         else:
    #             out += self.myfc(dropout(x))
    #     out /= len(self.dropout_fold)
    #     return out

    
    #インスタンスのクラス属性の表示
    def print_attribute(self,key=True,value=True):
      for i, j in self.__dict__.items():
        if key==True and value == True:
          print("#########key#########",i, "#########value#########", j)
        elif value==False: 
          print(i)
        else:
          print(j)

    #モデルサマリーgpuセットしないとエラー吐く→この関数実行するとcudaにセットされちゃう
    def model_summary(self):
        self = self.to(device)
        summary(self.arch,(3,256,256))
        # summary(self.meta,((10,1)))
        # summary(self.meta,(10))