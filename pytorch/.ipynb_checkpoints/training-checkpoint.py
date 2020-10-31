#トレーニング＋predict oof
import torch
# import torchvision
import torch.nn.functional as F
import torch.nn as nn
# import torchtoolbox.transform as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import pandas as pd
import numpy as np
print("クラス一覧")
print("Albu_Datset","Torch_Dataset")

def print_config():
    from pytorch import config
    # import importlib
    # importlib.reload(pytorch.config)
    print("DEBUG",config.DEBUG)
    print("image_size",config.image_size)
    print("epochs",config.epochs)
    print("es_patience",config.es_patience)
    print("batch_size",config.batch_size)
    print("num_workers",config.num_workers)
    print("TTA",config.TTA)
    print("kfold",config.kfold)
    print("save_path",config.save_path)
    print("b_num",config.b_num)
    print("train_aug,val_aug,test_aug",config.train_aug,config.val_aug,config.test_aug)

class Albu_Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, imfolder: str, phase:str = "train", 
                 transforms = None, aug = True, target="target"):
        self.df = df
        self.imfolder = imfolder
        self.transforms = transforms
        self.phase = phase
        self.aug = aug
        self.target = target
        
    def __getitem__(self, index, img_colmun="image_name", extension=".jpg"):
        #画像のパスから読み出し
        im_path = self.df.iloc[index][img_colmun] + extension
        img = cv2.imread(im_path)
        #albu用の処理(torchivisionではいらない)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #画像の前処理を実施
        img_transformed = self.transforms(img, self.aug)
        #albu用の処理(torchivisionではいらない)
        img_transformed = img_transformed.astype(np.float32)
        img_transformed = img_transformed.transpose(2, 0, 1)
        img_transformed = torch.tensor(img_transformed).float()
        #ラベルを取り出すか否か(テストかトレインか)
        if self.phase == "train":
            y = torch.tensor(self.df.iloc[index][self.target])
            return img_transformed,y
        else:
            return img_transformed

    def __len__(self):
        return len(self.df)


class Torch_Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, imfolder: str, phase:str = "train", 
                 transforms = None, aug = True, target="target"):
        self.df = df
        self.imfolder = imfolder
        self.transforms = transforms
        self.phase = phase
        self.aug = aug
        self.target = target
        
    def __getitem__(self, index, img_colmun="image_name", extension=".jpg"):
        #画像のパスから読み出し
        im_path = self.df.iloc[index][img_colmun] + extension
        img = cv2.imread(im_path)
        #albu用の処理(torchivisionではいらない)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #画像の前処理を実施
        img_transformed = self.transforms(img, self.aug)
        #ラベルを取り出すか否か(テストかトレインか)
        if self.phase == "train":
            y = torch.tensor(self.df.iloc[index][self.target])
            return img_transformed,y
        else:
            return img_transformed

    def __len__(self):
        return len(self.df)