#トレーニング＋predict oof
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import warnings,os,importlib
from tqdm import tqdm
#自作モジュール
from pytorch.dataset import Albu_Dataset
from pytorch.transform import Albu_Transform
from pytorch.model import Ef_Net

#コンフィグの読み直し！！
import pytorch
from pytorch import config
from pytorch import dataset
importlib.reload(pytorch.config)
importlib.reload(pytorch.dataset)
#predict_config
DEBUG = config.DEBUG
image_size = config.image_size
batch_size = config.batch_size
num_workers = config.num_workers
kfold = config.kfold
test_aug = config.test_aug
target = config.target
TTA = config.TTA
model_name = config.model_name
model_path = config.model_path
predict_path = config.predict_path
oof_path = config.oof_path
device = config.device
LOG_DIR, LOG_NAME = config.LOG_DIR, config.LOG_NAME
use_meta,out_features = config.use_meta,config.out_features

def get_trans(img, I):
    if I >= 4:
        img = img.transpose(2,3)
    if I % 4 == 0:
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)

def get_predict(df_test):
    print("device_CPU_GPU:", device)
    predict = torch.zeros((len(df_test), 1), dtype=torch.float32, device=device) 
    OUTPUTS = []
    test_dataset = Albu_Dataset(df=df_test, phase="test", transforms=Albu_Transform(image_size=image_size), aug=test_aug)
    #test_datasetの内容がfoldごとで変化がないのでここで読んでオーケイ
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    for fold in range(kfold):
        print('=' * 20, 'Fold', fold, '=' * 20)
        model_path_fold = model_path + model_name + "_fold{}.pth".format(fold)
        print(model_path_fold)
        model = Ef_Net()
        model = model.to(device)
        model.load_state_dict(torch.load(model_path_fold))
        model.eval()

        LOGITS = []
        PROBS = []
        bar = tqdm(test_loader,position=0,leave=True)
        with torch.no_grad():
            for (data) in bar:
                if use_meta:
                    data, meta = data
                    data, meta = data.to(device), meta.to(device)
                    logits = torch.zeros((data.shape[0], out_features)).to(device)
                    probs = torch.zeros((data.shape[0], out_features)).to(device)
                    for I in range(TTA):
                        l = model(get_trans(data, I), meta)
                        logits += l
                        probs += l.softmax(1)
                else:
                    data = data.to(device)
                    logits = torch.zeros((data.shape[0], out_features)).to(device)
                    probs = torch.zeros((data.shape[0], out_features)).to(device)
                    for I in range(TTA):
                        l = model(get_trans(data, I))
                        logits += l
                        # probs += l.softmax(1)
                        probs += l.sigmoid()
                logits /= TTA
                probs /= TTA
        
                LOGITS.append(logits.detach().cpu())
                PROBS.append(probs.detach().cpu())
                bar.set_description("get_predict")

        LOGITS = torch.cat(LOGITS).numpy()
        PROBS = torch.cat(PROBS).numpy()
        # OUTPUTS.append(PROBS[:, mel_idx])
        OUTPUTS.append(PROBS)
    pred = np.zeros(OUTPUTS[0].shape[0])
    for probs in OUTPUTS:
      probs = np.squeeze(probs)
      #rankにするか否か
      if False:
        pred += pd.Series(probs).rank(pct=True).values
      else:
        pred += pd.Series(probs).values
    pred /= len(OUTPUTS)
    df_test['target'] = pred
    return df_test
    
def predict_tocsv(df_sub, predict_temp):
    df_sub[target] = predict_temp.cpu().numpy().reshape(-1,)
    df_sub.to_csv(predict_path + "predict_{}.csv".format(model_name), index=False)

def run(df_test,df_sub,):
    from pytorch.seed import seed_everything
    seed_everything(1)
    warnings.simplefilter('ignore')
    test_predict = main(df_test, )
    predict_tocsv(df_sub, test_predict)
    return test_predict

if __name__ == '__main__':
    run(df_test,df_sub,imfolder_test)

#########使用方法##########
# import importlib
# from pytorch import predict
# import pytorch.predict
# importlib.reload(pytorch.predict)
# oof = predict.run(df_test,df_sub,imfolder_test)