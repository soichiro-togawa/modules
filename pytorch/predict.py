#トレーニング＋predict oof
import torch
from torch.utils.data import DataLoader
import numpy as np, pandas as pd
import os, sys, warnings, argparse
from tqdm import tqdm

#自作モジュール
from pytorch.dataset import Albu_Dataset
from pytorch.transform import Albu_Transform, get_trans
from pytorch.model import Ef_Net
from pytorch.seed import seed_everything
from pytorch import config


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
    
def predict_tocsv(df_test):
  df_test[['image_name', 'target']].to_csv('/content/submission.csv', index=False)

def run(df_test):
    warnings.simplefilter('ignore')
    seed_everything(1)
    df_test = get_predict(df_test)
    predict_tocsv(df_test)
    return df_test

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('arg1', help='df_train_path')
    #入力されたものだけ回収
    args, _ = parser.parse_known_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    df_test = pd.read_csv(args.arg1)
    run(df_test)

#########使用方法##########
# import importlib
# from pytorch import predict
# import pytorch.predict
# importlib.reload(pytorch.predict)
# oof = predict.run(df_test,df_sub,imfolder_test)