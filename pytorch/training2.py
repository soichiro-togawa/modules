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
from tqdm import tqdm
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
#追加のコンフィグ
output_dimout_features=
criterion=
init_lr

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
    logger.info("train_aug,val_aug:{}{}".format(train_aug,val_aug))
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

def train_epoch(model, loader, optimizer):
    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, target) in bar:
        optimizer.zero_grad()
        if use_meta:
            data, meta = data
            data, meta, target = data.to(device), meta.to(device), target.to(device)
            logits = model(data, meta)
        else:
            data, target = data.to(device), target.to(device)
            logits = model(data)
        loss = criterion(logits, target)

        if not use_amp:
            loss.backward()
        else:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

        optimizer.step()
        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))
    return train_loss

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

def val_epoch(model, loader, is_ext=None, n_test=1, get_output=False):
    model.eval()
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []
    with torch.no_grad():
        for (data, target) in tqdm(loader):
            
            if use_meta:
                data, meta = data
                data, meta, target = data.to(device), meta.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], out_dim)).to(device)
                probs = torch.zeros((data.shape[0], out_dim)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I), meta)
                    logits += l
                    probs += l.softmax(1)
            else:
                data, target = data.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], out_dim)).to(device)
                probs = torch.zeros((data.shape[0], out_dim)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I))
                    logits += l
                    probs += l.softmax(1)
            logits /= n_test
            probs /= n_test

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = criterion(logits, target)
            val_loss.append(loss.detach().cpu().numpy())

    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

    if get_output:
        return PROBS
    else:
        acc = (PROBS.argmax(1) == TARGETS).mean() * 100.
        auc = roc_auc_score((TARGETS==mel_idx).astype(float), PROBS[:, mel_idx])
        auc_20 = roc_auc_score((TARGETS[is_ext==0]==mel_idx).astype(float), PROBS[is_ext==0, mel_idx])
        return val_loss, acc, auc, auc_20

def run(fold):
    
    i_fold = fold

    if DEBUG:
        df_this = df_train[df_train['fold'] != i_fold].sample(batch_size * 3)
        df_valid = df_train[df_train['fold'] == i_fold].sample(batch_size * 3)
    else:
        df_this = df_train[df_train['fold'] != i_fold]
        df_valid = df_train[df_train['fold'] == i_fold]
    
    #データセット及びローダー
    dataset_train = Albu_Dataset(df=df_this,
                                phase="train", 
                                transforms=Albu_Transform(image_size=image_size),
                                aug=train_aug)
    dataset_valid = Albu_Dataset(df_valid,
                                phase="train", 
                                transforms=Albu_Transform(image_size=image_size),
                                aug=train_aug)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    #モデルの読み込み
    model = Ef_Net(n_meta_features=n_meta_features, out_features=out_features)
    model = model.to(device)
    logger.info("device_GPU_True:{}".format(next(model.parameters()).is_cuda))

    auc_max = 0.
    auc_20_max = 0.
    model_file = f'{kernel_type}_best_fold{i_fold}.pth'
    model_file2 = f'{kernel_type}_best_o_fold{i_fold}.pth'
    #オプティマイザー
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    if use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        
    #スケジューラ
    # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cosine_epo)
    # scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=warmup_epo, after_scheduler=scheduler_cosine)
    scheduler_warmup = ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=1, verbose=True, factor=0.2)

    print(len(dataset_train), len(dataset_valid))

    for epoch in range(1, n_epochs+1):
        print(time.ctime(), 'Epoch:', epoch)
        scheduler_warmup.step(epoch-1)

        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss, acc, auc, auc_20 = val_epoch(model, valid_loader, is_ext=df_valid['is_ext'].values)

        content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {np.mean(train_loss):.5f}, valid loss: {(val_loss):.5f}, acc: {(acc):.4f}, auc: {(auc):.6f}, auc_20: {(auc_20):.6f}.'
        print(content)
        with open(f'log_{kernel_type}.txt', 'a') as appender:
            appender.write(content + '\n')

        if auc > auc_max:
            print('auc_max ({:.6f} --> {:.6f}). Saving model ...'.format(auc_max, auc))
            torch.save(model.state_dict(), model_file)
            auc_max = auc
        if auc_20 > auc_20_max:
            print('auc_20_max ({:.6f} --> {:.6f}). Saving model ...'.format(auc_20_max, auc_20))
            torch.save(model.state_dict(), model_file2)
            auc_20_max = auc_20

    scores.append(auc_max)
    scores_20.append(auc_20_max)
    torch.save(model.state_dict(), os.path.join(f'{kernel_type}_model_fold{i_fold}.pth'))





def main(df_train, df_test):
    #ロガーのセット
    logger = setup_logger(LOG_DIR, LOG_NAME)
    print_config_log(logger)
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
        # criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.BCELoss()
        criterion = nn.CrossEntropyLoss()
        logger.info("criterion:{}".format(str(criterion)))

        #モデルクラスの読み込み
        model = Ef_Net()
        model = model.to(device)
        logger.info("device_GPU_True:{}".format(next(model.parameters()).is_cuda))

        #オプティマイザー、スケジューラ―のセット
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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
                x = torch.tensor(x, device=device, dtype=torch.float32)
                # print("is_cuda_x",x.is_cuda)
                # print(x.dtype)
                y = torch.tensor(y, device=device, dtype=torch.float32)
                # print("is_cuda_y",y.is_cuda)
            
                #順伝播
                optimizer.zero_grad()
                output = model(x)
                #損失(誤差)の計算
                #タクラス分類の場合
                y = y.long()
                # loss = criterion(output, y.unsqueeze(1))
                loss = criterion(output, y)

                #逆伝播
                #apex2
                # with amp.scale_loss(loss, optimizer) as scaled_loss:
                #     scaled_loss.backward()
                loss.backward()
                optimizer.step()

                #予測➡roundいるのかどうか問題
                # pred = torch.round(torch.sigmoid(output))
                # pred = torch.round(output)
                print(output)
                pred = output.softmax(1)
                print("pred",pred)

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
                    # val_pred = torch.sigmoid(output_val)
                    # val_pred = output_val
                    val_pred = output_val.softmax(1)

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

                # val_pred = torch.sigmoid(output_val)
                # val_pred = output_val
                val_pred = output_val.softmax(1)
                
                val_preds[j*val_loader.batch_size:j*val_loader.batch_size + x_val.shape[0]] = val_pred
            oof[val_idx] = val_preds.cpu().numpy()
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

if __name__ == '__main__':
    run()

#########使用方法##########
# import importlib
# from pytorch import training
# import pytorch.training
# importlib.reload(pytorch.training)
# oof = training.run(df_train, df_test,imfolder_train,imfolder_val)