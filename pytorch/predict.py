#インポート
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
from pytorch.branch_func import branch

#コンフィグ
DEBUG, image_size, batch_size, num_workers, kfold, test_aug,TTA\
= config.TRAIN_CONFIG["DEBUG"], config.TRAIN_CONFIG["image_size"], config.TRAIN_CONFIG["batch_size"], config.TRAIN_CONFIG["num_workers"], config.TRAIN_CONFIG["kfold"], config.TRAIN_CONFIG["test_aug"], config.TRAIN_CONFIG["TTA"]
#既出
use_meta, target, out_features, device, criterion\
= config.TRAIN_CONFIG["use_meta"], config.TRAIN_CONFIG["target"], config.TRAIN_CONFIG["out_features"], config.TRAIN_CONFIG["device"], config.TRAIN_CONFIG["criterion"]
#path
model_path, predict_path, LOG_DIR, LOG_NAME\
= config.PATH_CONFIG["model_path"], config.PATH_CONFIG["predict_path"], config.PATH_CONFIG["LOG_DIR"], config.PATH_CONFIG["LOG_NAME"]

#前段階=lossの種類による分岐
func1,func2,func3,func4,func5,func6 = branch(criterion)

#関数
def get_predict(df_test,mel_idx=6):
    print("device_CPU_GPU:", device)
    predict = torch.zeros((len(df_test), 1), dtype=torch.float32, device=device) 
    OUTPUTS = []
    test_dataset = Albu_Dataset(df=df_test, phase="test", transforms=Albu_Transform(image_size=image_size), aug=test_aug, use_meta=use_meta)
    #test_datasetの内容がfoldごとで変化がないのでここで読んでオーケイ
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    for fold in range(kfold):
        print('=' * 20, 'Fold', fold, '=' * 20)
        model_path_fold_1 = model_path + "_fold{}_1.pth".format(fold)
        print(model_path_fold_1)
        n_meta_features =  test_dataset.n_meta_features #無理やり組み込んだ
        model = Ef_Net(n_meta_features=n_meta_features, out_features=out_features)
        model = model.to(device)
        model.load_state_dict(torch.load(model_path_fold_1))
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
                        probs = func1(probs,l)
                else:
                    data = data.to(device)
                    logits = torch.zeros((data.shape[0], out_features)).to(device)
                    probs = torch.zeros((data.shape[0], out_features)).to(device)
                    for I in range(TTA):
                        l = model(get_trans(data, I))
                        logits += l
                        probs = func1(probs,l)
                logits /= TTA
                probs /= TTA
        
                LOGITS.append(logits.detach().cpu())
                PROBS.append(probs.detach().cpu())
                bar.set_description("get_predict")

        LOGITS = torch.cat(LOGITS).numpy()
        PROBS = torch.cat(PROBS).numpy()
        func6(OUTPUTS,PROBS,mel_idx)
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
  df_test[['image_name', 'target']].to_csv(predict_path, index=False)

def run(df_test):
    warnings.simplefilter('ignore')
    seed_everything(1)
    df_test = get_predict(df_test)
    predict_tocsv(df_test)
    return df_test

#実行関数
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