#データセット
import numpy as np
import pandas as pd
import torch.tensor
from torch.utils.data import Dataset
import cv2
print("クラス一覧")
print("Albu_Datset","Torch_Dataset")


class Albu_Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, phase:str = "train", 
                 transforms = None, aug = True, target="target"):
        self.df = df
        self.transforms = transforms
        self.phase = phase
        self.aug = aug
        self.target = target
        
    def __getitem__(self, index, img_column="image_name", extension=""):
        #画像のパスから読み出し
        im_path = self.df.iloc[index][img_column] + extension
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
    def __init__(self, df: pd.DataFrame, phase:str = "train", 
                 transforms = None, aug = True, target="target"):
        self.df = df
        self.transforms = transforms
        self.phase = phase
        self.aug = aug
        self.target = target
        
    def __getitem__(self, index, img_column="image_name", extension=""):
        #画像のパスから読み出し
        im_path = self.df.iloc[index][img_column] + extension
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

#########使用方法##########
# from pytorch.dataset import Albu_Dataset

# from sklearn.model_selection import train_test_split
# #分割対象をインデックスに
# train_idx, val_idx = train_test_split(train_df.index, test_size=0.2, shuffle=True, random_state=1,stratify=df_train["target"])
# print(df_train.iloc[train_idx,-1].value_counts())
# print(df_train.iloc[val_idx,-1].value_counts())