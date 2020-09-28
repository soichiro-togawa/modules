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
import time, datetime, warnings,os
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
#emsamble_config
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

def print_config():
    print("DEBUG",DEBUG)
    print("image_size",image_size)
    print("epochs",epochs)
    print("es_patience",es_patience)
    print("batch_size",batch_size)
    print("num_workers",num_workers)
    print("kfold",kfold)
    print("b_num",b_num)
    print("train_aug,val_aug",train_aug,val_aug)
    print("--------path----------")
    print("model_name",model_name)
    print("model_path",model_path,os.path.exists(model_path))
    print("predict_path",predict_path,os.path.exists(predict_path))
    print("oof_path",oof_path,os.path.exists(oof_path))
    if DEBUG==True:
      print("#########DEBUG-MODE#############")
      print("model_path",os.path.exists(config.temp_model_path))
      print("oof_path",os.path.exists(config.temp_oof_path))
      print("predict_path",os.path.exists(config.temp_predict_path))

#コンフィグのログ
def print_config_log(logger):
    logger.info("#####Print_Config######")
    logger.info("DEBUG:{}".format(DEBUG))
    logger.info("image_size:{}".format(image_size))
    logger.info("epochs:{}".format(epochs))
    logger.info("es_patience:{}".format(es_patience))
    logger.info("batch_size:{}".format(batch_size))
    logger.info("num_workers:{}".format(num_workers))
    logger.info("kfold:{}".format(kfold))
    logger.info("b_num:{}".format(b_num))
    logger.info("train_aug,val_aug:{}".format(train_aug,val_aug))
    logger.info("--------path----------")
    logger.info("model_name:{}".format(model_name))
    logger.info("model_path:{}:{}".format(model_path,os.path.exists(model_path)))
    logger.info("predict_path:{}:{}".format(predict_path,os.path.exists(predict_path)))
    logger.info("oof_path:{}:{}".format(oof_path,os.path.exists(oof_path)))
    if DEBUG==True:
      logger.info("#########DEBUG-MODE#############")
      logger.info("model_path:{}".format(os.path.exists(config.temp_model_path)))
      logger.info("oof_path:{}".format(os.path.exists(config.temp_oof_path)))
      logger.info("predict_path:{}".format(os.path.exists(config.temp_predict_path)))
    logger.info("#####END######")


def main(df_train, df_test, imfolder_train,imfolder_val):
    #ロガーのセット
    logger = setup_logger(LOG_DIR, LOG_NAME)
    print_config_log(logger)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device_CPU_GPU:{}".format(device))
    oof = np.zeros((len(df_train), 1))  # Out Of Fold predictions
    preds = torch.zeros((len(df_test), 1), dtype=torch.float32, device=device) 
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=1)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X=np.zeros(len(df_train)), y=df_train[target])):
        logger.info("{0}Fold{1}{2}".format("="*20,fold,"="*20))
        model_path_fold = model_path + model_name + "_fold{}.pth".format(fold)  # Path and filename to save model to
        best_val = 0  # Best validation score within this fold
        patience = es_patience  # Current patience counter
        #モデルクラスの読み込み
        model = Ef_Net()
        model = model.to(device)
        # logger.info("device_GPU_True:{}".format(next(model.parameters()).is_cuda))
        #オプティマイザー、スケジューラ―、損失関数
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=1, verbose=True, factor=0.2)
        criterion = nn.BCEWithLogitsLoss()
        
        #データセットインスタンスの生成
        train_dataset = Albu_Dataset(df=df_train.iloc[train_idx].reset_index(drop=True), 
                                imfolder=imfolder_train, 
                                phase="train", 
                                transforms=Albu_Transform(image_size=image_size),
                                aug=train_aug)
        val_dataset = Albu_Dataset(df=df_train.iloc[val_idx].reset_index(drop=True), 
                                imfolder=imfolder_val, 
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
                x = torch.tensor(x, device=device, dtype=torch.float32)
                # print("is_cuda_x",x.is_cuda)
                # print(x.dtype)
                y = torch.tensor(y, device=device, dtype=torch.float32)
                # print("is_cuda_y",y.is_cuda)
            
                #順伝播
                optimizer.zero_grad()
                output = model(x)
                #損失(誤差)の計算
                loss = criterion(output, y.unsqueeze(1))
                # loss = criterion(output, y)

                #逆伝播
                #apex2
                # with amp.scale_loss(loss, optimizer) as scaled_loss:
                #     scaled_loss.backward()
                loss.backward()
                optimizer.step()

                #予測          
                pred = torch.round(torch.sigmoid(output))  # round oftain predictionsf sigmoid to ob
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

                    val_preds[j*val_loader.batch_size:j*val_loader.batch_size + x_val.shape[0]] = val_pred
                val_acc = accuracy_score(df_train.iloc[val_idx][target].values, torch.round(val_preds.cpu()))
                val_roc = roc_auc_score(df_train.iloc[val_idx][target].values, val_preds.cpu())
                
                #表示
                # print('Epoch {:03}: | Loss: {:.3f} | Train acc: {:.3f} | Val acc: {:.3f} | Val roc_auc: {:.3f} | Training time: {}'.format(
                # epoch + 1, 
                # epoch_loss, 
                # train_acc, 
                # val_acc, 
                # val_roc, 
                # str(datetime.timedelta(seconds=time.time() - start_time))[:7]))
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
                    torch.save(model, model_path_fold)  # Saving current best model
                else:
                    patience -= 1
                    if patience == 0:
                        print('Early stopping. Best Val roc_auc: {:.3f}'.format(best_val))
                        logger.info('Early stopping. Best Val roc_auc: {:.3f}'.format(best_val))
                        break
                    
        model = torch.load(model_path_fold)  # Loading best model of this fold
        model.eval()  # switch model to the evaluation mode
        val_preds = torch.zeros((len(val_idx), 1), dtype=torch.float32, device=device)
        with torch.no_grad():
            # Predicting on validation set once again to obtain data for OOF
            for j, (x_val, y_val) in enumerate(val_loader):
                x_val = torch.tensor(x_val, device=device, dtype=torch.float32)
                y_val = torch.tensor(y_val, device=device, dtype=torch.float32)
                output_val = model(x_val)
                val_pred = torch.sigmoid(output_val)
                val_preds[j*val_loader.batch_size:j*val_loader.batch_size + x_val.shape[0]] = val_pred
            oof[val_idx] = val_preds.cpu().numpy()
    return oof

def oof_tocsv(oof_temp):
    oof_temp = pd.Series(oof_temp.reshape(-1,))
    oof_temp.to_csv(oof_path + "oof_{}.csv".format(model_name), index=False)

def run(df_train, df_test, imfolder_train, imfolder_val):
    from pytorch.seed import seed_everything
    seed_everything(1)
    warnings.simplefilter('ignore')
    oof = main(df_train, df_test, imfolder_train,imfolder_val)
    oof_tocsv(oof)
    return oof

if __name__ == '__main__':
    run()

#########使用方法##########
# import importlib
# from pytorch import training
# import pytorch.training
# importlib.reload(pytorch.training)
# oof = training.run(df_train, df_test,imfolder_train,imfolder_val)