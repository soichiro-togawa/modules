#seedの固定
import os,random
import numpy as np
import torch

def seed_everything(seed=1,deterministic=True,benchmark=True):
    #hash値の環境変数を新規作成
    os.environ['PYTHONHASHSEED'] = str(seed)
    #pythonの乱数生成:RNG (Random Number Generator:乱数ジェネレータ)
    random.seed(seed)
    #numpyの乱数生成
    np.random.seed(seed)
    #pytorchの乱数生成
    torch.manual_seed(seed)
    #cudaの乱数生成（GPU使用時のみ)
    torch.cuda.manual_seed(seed)
    #Trueでロスが同じ結果になる？
    torch.backends.cudnn.deterministic = deterministic
    #動的なモデル(特定の条件で分岐するようなモデルの場合)はFalseにすべき
    torch.backends.cudnn.benchmark = benchmark
    print("乱数設定完了")

#########使用方法##########
# from pytorch.seed import seed_everything
# seed_everything(1)