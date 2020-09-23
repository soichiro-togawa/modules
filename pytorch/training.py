#トレーニング＋predict oof
from pytorch.dataset import Albu_Dataset
from pytorch.transform import Albu_Transform
import torch
# import torchvision
import torch.nn.functional as F
import torch.nn as nn
# import torchtoolbox.transform as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import pandas as pd
import numpy as np
import time, datetime, warnings


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
save_path = config.save_path
b_num = config.b_num
train_aug, val_aug = config.train_aug, config.val_aug
if DEBUG == True:
  epochs = 2
  kfold = 2
  save_path = save_path + "_DEBUG"

def print_config():
    print("DEBUG",DEBUG)
    print("image_size",image_size)
    print("epochs",epochs)
    print("es_patience",es_patience)
    print("batch_size",batch_size)
    print("num_workers",num_workers)
    print("kfold",kfold)
    print("save_path",save_path)
    print("b_num",b_num)
    print("train_aug,val_aug",train_aug,val_aug)
    if DEBUG==True:
      print("#########DEBUG-MODE#############")
print_config()


def main(df_train, df_test, imfolder_train,imfolder_val):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device_CPU_GPU:", device)
    oof = np.zeros((len(df_train), 1))  # Out Of Fold predictions
    preds = torch.zeros((len(df_test), 1), dtype=torch.float32, device=device) 
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=1)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X=np.zeros(len(df_train)), y=df_train['target'])):
        print('=' * 20, 'Fold', fold, '=' * 20)
        model_path = save_path + f'model_fold_{fold}.pth'  # Path and filename to save model to
        best_val = 0  # Best validation score within this fold
        patience = es_patience  # Current patience counter

        #モデル定義
        # in_features = {"b0":1000,"b1":1280,"b2":1408,"b3":1536,"b4":1792,"b5":2048,"b6":2304,"b7":2560}
        model = EfficientNet.from_pretrained('efficientnet-'+b_num)

        # Unfreeze model weights→凍結せず
        for param in model.parameters():
            param.requires_grad = True
        num_ftrs = model._fc.in_features
        # print(num_ftrs)
        model._fc = nn.Linear(in_features=num_ftrs, out_features=1)
        model = model.to(device)
        
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
                # print(x.dtype)
                y = torch.tensor(y, device=device, dtype=torch.float32)
            
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
                val_acc = accuracy_score(df_train.iloc[val_idx]['target'].values, torch.round(val_preds.cpu()))
                val_roc = roc_auc_score(df_train.iloc[val_idx]['target'].values, val_preds.cpu())
                
                #表示
                print('Epoch {:03}: | Loss: {:.3f} | Train acc: {:.3f} | Val acc: {:.3f} | Val roc_auc: {:.3f} | Training time: {}'.format(
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
                    torch.save(model, model_path)  # Saving current best model
                else:
                    patience -= 1
                    if patience == 0:
                        print('Early stopping. Best Val roc_auc: {:.3f}'.format(best_val))
                        break
                    
        model = torch.load(model_path)  # Loading best model of this fold
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

def run(df_train, df_test, imfolder_train, imfolder_val):
    from pytorch.seed import seed_everything
    seed_everything(1)
    warnings.simplefilter('ignore')
    oof = main(df_train, df_test, imfolder_train,imfolder_val)
    return oof

if __name__ == '__main__':
    run()

