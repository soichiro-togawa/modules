#インポート
import numpy as np, pandas as pd
import torch.tensor
from torch.utils.data import Dataset
import cv2

class Albu_Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, phase:str = "train", 
                 transforms = None, aug = True, target="target", use_meta=False):
        self.df = df
        self.transforms = transforms
        self.phase = phase
        self.aug = aug
        self.target = target
        self.use_meta = use_meta
        #メタ使う場合→df_train["meta"]に無理やり格納している
        if self.use_meta:
            #インデックスふり直してないから注意
            # temp = self.df["meta"][self.df.index[0]] #こっちでも可能
            temp = self.df.iloc[0,:]["meta"]
            temp = temp.split(",")
            self.meta_features = temp[:-1]
            self.n_meta_features = int(temp[-1])
            temp2 = self.df.iloc[0,:]["target_index"]
            self.target_index = int(temp2) 
        else:
            self.n_meta_features =0

    def __getitem__(self, index, img_column="image_path", extension=""):
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
        if self.use_meta:
            img_transformed = (img_transformed, torch.tensor(self.df.iloc[index][self.meta_features]).float())

        #ラベルを取り出すか否か(テストかトレインか)
        if self.phase == "train":
            y = torch.tensor(self.df.iloc[index][self.target])
            # y = y.float()
            y = y.long()
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
        
    def __getitem__(self, index, img_column="image_path", extension=""):
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