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
import time, datetime, warnings, os,sys, argparse,importlib
from tqdm import tqdm

#自作モジュール
from pytorch.dataset import Albu_Dataset
from pytorch.transform import Albu_Transform
from pytorch.model import Ef_Net
from pytorch.seed import seed_everything
from pytorch import config
from make_log import setup_logger

#コンフィグの読み直し！！
import pytorch
importlib.reload(pytorch)
# importlib.reload(pytorch.model)
# importlib.reload(pytorch.dataset)

#config_import_list
DEBUG, image_size, epochs, es_patience, batch_size, num_workers, kfold, target, b_num, train_aug, val_aug\
= config.DEBUG, config.image_size, config.epochs, config.es_patience, config.batch_size, config.num_workers, config.kfold, config.target, config.b_num, config.train_aug, config.val_aug
model_name, model_path, predict_path, oof_path,LOG_DIR, LOG_NAME,USE_AMP, criterion, out_features, use_meta\
= config.model_name, config.model_path, config.predict_path, config.oof_path, config.LOG_DIR, config.LOG_NAME, config.USE_AMP, config.criterion,config. out_features, config.use_meta
n_val, device = config.n_val, config.device

if USE_AMP==True:
  from apex import amp

def train_epoch(model, loader, optimizer,logger):
    model.train()
    train_loss = []
    bar = tqdm(loader,position=0,leave=True)
    for (data, target) in bar:
        optimizer.zero_grad()
        if use_meta:
            data, meta = data
            data, meta, target = data.to(device), meta.to(device), target.to(device)
            logits = model(data, meta)
        else:
            data, target = data.to(device), target.to(device)
            logits = model(data)
        # loss = criterion(logits, target)
        loss = criterion(logits, target.unsqueeze(1))

        if not USE_AMP:
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
        
def val_epoch(model, loader, is_ext=None, n_test=1, get_output=False,mel_idx=1):
    model.eval()
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []
    bar = tqdm(loader,position=0,leave=True)
    with torch.no_grad():
        for (data, target) in bar:
            if use_meta:
                data, meta = data
                data, meta, target = data.to(device), meta.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], out_features)).to(device)
                probs = torch.zeros((data.shape[0], out_features)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I), meta)
                    logits += l
                    # probs += l.softmax(1)
                    probs += l.sigmoid()
            else:
                data, target = data.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], out_features)).to(device)
                probs = torch.zeros((data.shape[0], out_features)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I))
                    logits += l
                    # probs += l.softmax(1)
                    probs += l.sigmoid()
            logits /= n_test
            probs /= n_test

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())

            # loss = criterion(logits, target)
            loss = criterion(logits, target.unsqueeze(1))
            val_loss.append(loss.detach().cpu().numpy())
            if get_output:
              bar.set_description("get_oof")
            else:
              bar.set_description('val_loss: %.5f' % (loss))

    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

    if get_output:
        return PROBS
    else:
        acc = (PROBS.argmax(1) == TARGETS).mean() * 100.
        # auc = roc_auc_score((TARGETS==mel_idx).astype(float), PROBS[:, mel_idx])
        auc = roc_auc_score(TARGETS.astype(float), PROBS)
        # auc_20 = roc_auc_score((TARGETS[is_ext==0]==mel_idx).astype(float), PROBS[is_ext==0, mel_idx])
        auc_20 = roc_auc_score((TARGETS[is_ext==0]).astype(float), PROBS[is_ext==0])
        return val_loss, acc, auc, auc_20

def get_oof(df_train):
  PROBS = []
  oof = []
  for fold in range(kfold):
    df_valid = df_train[df_train["val_fold"] == fold]
    val_dataset = Albu_Dataset(df=df_valid, 
                                phase="train", 
                                transforms=Albu_Transform(image_size=image_size),
                                aug=val_aug)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model_path_fold_1 = model_path + model_name + "_fold{}_1.pth".format(fold)
    model = Ef_Net()
    model = model.to(device)
    model.load_state_dict(torch.load(model_path_fold_1))
    model.eval()
    this_PROBS = val_epoch(model, val_loader, is_ext=df_valid['is_ext'].values, n_test=n_val, get_output=True)
    PROBS.append(this_PROBS)
    oof.append(df_valid)
  oof = pd.concat(oof).reset_index(drop=True)
  # oof['pred'] = np.concatenate(PROBS).squeeze()[:, mel_idx]
  oof['pred'] = np.concatenate(PROBS).squeeze()
  return oof


def get_fold(df_train):
  logger = setup_logger(LOG_DIR, LOG_NAME)
  config.log_config(logger)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  logger.info("device_CPU_GPU:{}".format(device))
  scores = []
  scores_20 = []
  PROBS = []
  dfs = []
  skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=1)
  for fold, (train_idx, val_idx) in enumerate(skf.split(X=np.zeros(len(df_train)), y=df_train[target])):
    df_train.loc[val_idx,"val_fold"] = fold

  for fold in range(kfold):
    logger.info("{0}Fold{1}{2}".format("="*20,fold,"="*20))
    model_path_fold_1 = model_path + model_name + "_fold{}_1.pth".format(fold)  # Path and filename to save model to
    model_path_fold_2 = model_path + model_name + "_fold{}_2.pth".format(fold)  # Path and filename to save model to
    best_val = 0  # Best validation score within this fold
    patience = es_patience  # Current patience counter
    val_auc_max = 0.
    val_auc_20_max = 0.

    #損失関数のセット
    # criterion = criterion
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
    
    df_this = df_train[df_train["val_fold"] != fold]
    df_valid = df_train[df_train["val_fold"] == fold]
    #データセットインスタンスの生成
    train_dataset = Albu_Dataset(df=df_this, 
                            phase="train", 
                            transforms=Albu_Transform(image_size=image_size),
                            aug=train_aug)
    val_dataset = Albu_Dataset(df=df_valid, 
                            phase="train", 
                            transforms=Albu_Transform(image_size=image_size),
                            aug=val_aug)
    #データローダーにセッット
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    #エポック
    for epoch in range(epochs):
      logger.info("{0} Epoch:{1}".format(time.ctime(),  epoch))
      train_loss = train_epoch(model, train_loader, optimizer,logger)
      val_loss, acc, val_auc, val_auc_20 = val_epoch(model, val_loader, is_ext=df_valid['is_ext'].values)
      content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {np.mean(train_loss):.5f}, valid loss: {(val_loss):.5f}, acc: {(acc):.4f}, val_auc: {(val_auc):.6f}, val_auc_20: {(val_auc_20):.6f}.'
      logger.info(content)
      
      #オプティマイザーはバッチ単位でstepする、スケジューラ―はエポック単位のためこのタイミング
      scheduler.step(val_auc)
      if val_auc<=val_auc_max and val_auc_20<=val_auc_20_max:
        patience -= 1
        if patience == 0:
            logger.info('Early stopping. Best Val roc_auc: {:.3f}'.format(val_auc_max))
            break

      if val_auc > val_auc_max:
          logger.info('val_auc_max ({:.6f} --> {:.6f}). Saving model ...'.format(val_auc_max, val_auc))
          val_auc_max = val_auc
          patience = es_patience  # Resetting patience since we have new best validation accuracy
          torch.save(model.state_dict(), model_path_fold_1)  # Saving current best model
      if val_auc_20 > val_auc_20_max:
          logger.info('val_auc_20_max ({:.6f} --> {:.6f}). Saving model ...'.format(val_auc_20_max, val_auc_20))
          torch.save(model.state_dict(), model_path_fold_2)
          val_auc_20_max = val_auc_20

  
    scores.append(val_auc_max)
    scores_20.append(val_auc_20_max)
    # torch.save(model.state_dict(), os.path.join(f'{kernel_type}_model_fold{i_fold}.pth'))
    
  #getoof
  oof = get_oof(df_train)
  logger.info("train_finish")
  return oof

def oof_tocsv(oof):
  oof.to_csv('/content/oof.csv', index=False)

def run(df_train):
    warnings.simplefilter('ignore')
    seed_everything(1)
    oof = get_fold(df_train)
    oof_tocsv(oof)
    return oof

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('arg1', help='df_train_path')
    #入力されたものだけ回収
    args, _ = parser.parse_known_args()
    return args
    
if __name__ == '__main__':
    args = parse_args()
    df_train = pd.read_csv(args.arg1)
    run(df_train)