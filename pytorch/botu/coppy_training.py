#トレーニング＋predict oof
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import time, datetime, warnings, os, argparse
# from apex import amp

import sys
sys.path.append("./")
# from dir1.mod1 import *
from make_log import setup_logger

#自作モジュール
from pytorch.dataset import Albu_Dataset
from pytorch.transform import Albu_Transform
from pytorch.model import Ef_Net

#コンフィグの読み直し！！
from pytorch import config
import pytorch
import importlib
importlib.reload(pytorch.config)
import pytorch.model
importlib.reload(pytorch.model)
import pytorch.dataset
importlib.reload(pytorch.dataset)
#config_import_list
DEBUG = config.DEBUG
image_size = config.image_size
epochs = config.epochs
es_patience = config.es_patience
batch_size = config.batch_size
num_workers = config.num_workers
kfold = config.kfold
target = config.target
b_num = config.b_num
train_aug, val_aug = config.train_aug, config.val_aug
model_name = config.model_name
model_path = config.model_path
predict_path = config.predict_path
oof_path = config.oof_path
LOG_DIR = config.LOG_DIR
LOG_NAME = config.LOG_NAME
USE_AMP = config.USE_AMP






def main(df_train, df_test):
    #ロガーのセット
    logger = setup_logger(LOG_DIR, LOG_NAME)
    config.log_config(logger)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device_CPU_GPU:{}".format(device))
    oof = np.zeros((len(df_train), 1))  # Out Of Fold predictions
    preds = torch.zeros((len(df_test), 1), dtype=torch.float32, device=device) 
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=1)

    #skf=5→インデックスをリストに格納して返す
    for fold, (train_idx, val_idx) in enumerate(skf.split(X=np.zeros(len(df_train)), y=df_train[target])):
        logger.info("{0}Fold{1}{2}".format("="*20,fold,"="*20))
        model_path_fold = model_path + model_name + "_fold{}.pth".format(fold)  # Path and filename to save model to
        best_val = 0  # Best validation score within this fold
        patience = es_patience  # Current patience counter

        #損失関数のセット
        criterion = nn.BCEWithLogitsLoss()
        logger.info("criterion:{}".format(str(criterion)))

        #モデルクラスの読み込み
        model = Ef_Net()
        model = model.to(device)
        logger.info("device_GPU_True:{}".format(next(model.parameters()).is_cuda))

        #オプティマイザー、スケジューラ―のセット
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        if USE_AMP:
          model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=1, verbose=True, factor=0.2)
        
        #データセットインスタンスの生成
        train_dataset = Albu_Dataset(df=df_train.iloc[train_idx].reset_index(drop=True), 
                                phase="train", 
                                transforms=Albu_Transform(image_size=image_size),
                                aug=train_aug)
        val_dataset = Albu_Dataset(df=df_train.iloc[val_idx].reset_index(drop=True), 
                                phase="train", 
                                transforms=Albu_Transform(image_size=image_size),
                                aug=val_aug)
        #データローダーにセッット
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        #ループ2(エポック単位)
        for epoch in range(epochs):
            start_time = time.time()
            #初期化
            correct = 0
            epoch_loss = 0
            #学習モードに切り替え
            model.train()
            #ループ処理3(ミニバッチ／ローダー単位)
            for x, y in train_loader:
                # print(x.dtype)
                #inputのxやlabelのyをcudaに乗せる+tensor,float32型にしている
                # x = torch.tensor(x, device=device, dtype=torch.float32)
                x = x.to(device)
                # print("is_cuda_x",x.is_cuda)
                # print(x.dtype)
                # y = torch.tensor(y, device=device, dtype=torch.float32)
                y = y.to(device)
                # print("is_cuda_y",y.is_cuda)
            
                #順伝播
                optimizer.zero_grad()
                output = model(x)
                #損失(誤差)の計算
                #タクラス分類の場合
                # y = y.long()
                loss = criterion(output, y.unsqueeze(1))
                # loss = criterion(output, y)

                #逆伝播
                if not USE_AMP:
                  loss.backward()
                else:
                  with amp.scale_loss(loss, optimizer) as scaled_loss:
                      scaled_loss.backward()

                optimizer.step()

                #予測➡roundいるのかどうか問題
                pred = torch.round(torch.sigmoid(output))
                # pred = torch.round(output)
                # pred = output.softmax(1)

                correct += (pred.cpu() == y.cpu().unsqueeze(1)).sum().item()  # tracking number of correctly predicted samples
                epoch_loss += loss.item()
            train_acc = correct / len(train_idx)
            
            #予測モードに切り替え→val
            model.eval()  # switch model to the evaluation mode
            val_preds = torch.zeros((len(val_idx), 1), dtype=torch.float32, device=device)
            with torch.no_grad():  # Do not calculate gradient since we are only predicting
                # Predicting on validation set
                for j, (x_val, y_val) in enumerate(val_loader):
                    x_val = torch.tensor(x_val, device=device, dtype=torch.float32)
                    y_val = torch.tensor(y_val, device=device, dtype=torch.float32)
                    output_val = model(x_val)
                    val_pred = torch.sigmoid(output_val)
                    # val_pred = output_val
                    # val_pred = output_val.softmax(1)

                    val_preds[j*val_loader.batch_size:j*val_loader.batch_size + x_val.shape[0]] = val_pred
                val_acc = accuracy_score(df_train.iloc[val_idx][target].values, torch.round(val_preds.cpu()))
                val_roc = roc_auc_score(df_train.iloc[val_idx][target].values, val_preds.cpu())
                
                #ログ抽出
                logger.info('Epoch {:03}: | Loss: {:.3f} | Train acc: {:.3f} | Val acc: {:.3f} | Val roc_auc: {:.3f} | Training time: {}'.format(
                epoch + 1, 
                epoch_loss, 
                train_acc, 
                val_acc, 
                val_roc, 
                str(datetime.timedelta(seconds=time.time() - start_time))[:7]))
                
                #オプティマイザーはバッチ単位でstepする、スケジューラ―はエポック単位のためこのタイミング
                scheduler.step(val_roc)
                    
                if val_roc >= best_val:
                    best_val = val_roc
                    patience = es_patience  # Resetting patience since we have new best validation accuracy
                    torch.save(model.state_dict(), model_path_fold)  # Saving current best model
                else:
                    patience -= 1
                    if patience == 0:
                        print('Early stopping. Best Val roc_auc: {:.3f}'.format(best_val))
                        logger.info('Early stopping. Best Val roc_auc: {:.3f}'.format(best_val))
                        break
                    
        model = Ef_Net()
        model.load_state_dict(torch.load(model_path_fold))
        model = model.to(device)
        model.eval()
        val_preds = torch.zeros((len(val_idx), 1), dtype=torch.float32, device=device)
        with torch.no_grad():
            # Predicting on validation set once again to obtain data for OOF
            for j, (x_val, y_val) in enumerate(val_loader):
                x_val = torch.tensor(x_val, device=device, dtype=torch.float32)
                y_val = torch.tensor(y_val, device=device, dtype=torch.float32)
                output_val = model(x_val)

                val_pred = torch.sigmoid(output_val)
                # val_pred = output_val
                # val_pred = output_val.softmax(1)
                
                val_preds[j*val_loader.batch_size:j*val_loader.batch_size + x_val.shape[0]] = val_pred
            oof[val_idx] = val_preds.cpu().numpy()
    logger.info("train_finish")
    return oof

def oof_tocsv(oof_temp):
    oof_temp = pd.Series(oof_temp.reshape(-1,))
    oof_temp.to_csv(oof_path + "oof_{}.csv".format(model_name), index=False)

def run(df_train, df_test):
    from pytorch.seed import seed_everything
    seed_everything(1)
    warnings.simplefilter('ignore')
    oof = main(df_train, df_test)
    oof_tocsv(oof)
    return oof

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('arg1', help='df_train_path')
    parser.add_argument('arg2', help='df_test_path')
    #入力されたものだけ回収
    args, _ = parser.parse_known_args()
    return args

# プロファイラーの使い方
# %cd "/content/drive/My Drive/Pipeline" 
# train_df.to_csv("/content/train_df.csv",index=False)
# test_df.to_csv("/content/test_df.csv",index=False)
# !nvprof -o profile.nvp \
#       python "pytorch/training.py" "/content/train_df.csv" "/content/test_df.csv"
if __name__ == '__main__':
    args = parse_args()
    df_train = pd.read_csv(args.arg1)
    df_test = pd.read_csv(args.arg2)
    run(df_train, df_test)

#########使用方法##########
# import importlib
# from pytorch import training
# import pytorch.training
# importlib.reload(pytorch.training)
# oof = training.run(df_train, df_test,imfolder_train,imfolder_val)

##pip###
# #apexのインストール
# !git clone https://github.com/NVIDIA/apex
# !pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex