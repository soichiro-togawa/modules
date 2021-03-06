#インポート
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
import numpy as np, pandas as pd
import os, sys, time, datetime, warnings, argparse
from tqdm import tqdm
#自作モジュール
from pytorch.dataset import Albu_Dataset
from pytorch.transform import Albu_Transform, get_trans
from pytorch.model import Ef_Net
from pytorch.seed import seed_everything
from pytorch import config
from pytorch.branch_func import branch
from make_log.make_log import setup_logger

#コンフィグ
DEBUG, USE_AMP, image_size, epochs, es_patience, batch_size, num_workers, kfold, fold_list, train_aug, val_aug, n_val\
= config.TRAIN_CONFIG["DEBUG"], config.TRAIN_CONFIG["USE_AMP"], config.TRAIN_CONFIG["image_size"], config.TRAIN_CONFIG["epochs"], config.TRAIN_CONFIG["es_patience"], config.TRAIN_CONFIG["batch_size"], config.TRAIN_CONFIG["num_workers"], config.TRAIN_CONFIG["kfold"], config.TRAIN_CONFIG["fold_list"], config.TRAIN_CONFIG["train_aug"], config.TRAIN_CONFIG["val_aug"], config.TRAIN_CONFIG["n_val"]
#既出
use_meta, target, out_features, device, criterion\
= config.TRAIN_CONFIG["use_meta"], config.TRAIN_CONFIG["target"], config.TRAIN_CONFIG["out_features"], config.TRAIN_CONFIG["device"], config.TRAIN_CONFIG["criterion"]
#path
model_path, oof_path, LOG_DIR, LOG_NAME\
= config.PATH_CONFIG["model_path"], config.PATH_CONFIG["oof_path"], config.PATH_CONFIG["LOG_DIR"], config.PATH_CONFIG["LOG_NAME"]
#apexのインポート
if USE_AMP:
    from apex import amp, optimizers

#前段階=lossの種類による分岐
func0,func1,func2,func3,func4,func5,func6 = branch(criterion)

#関数
def get_val_fold(df_train):
  skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=1)
  for fold, (train_idx, val_idx) in enumerate(skf.split(X=np.zeros(len(df_train)), y=df_train[target])):
    df_train.loc[val_idx,"val_fold"] = fold
  return df_train

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
        loss = func2(logits, target)

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

def val_epoch(model, loader, is_ext=None, n_test=1, get_output=False,target_index=6):
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
                    probs = func1(probs,l)
            else:
                data, target = data.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], out_features)).to(device)
                probs = torch.zeros((data.shape[0], out_features)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I))
                    logits += l
                    probs = func1(probs,l)
            logits /= n_test
            probs /= n_test

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = func2(logits, target)
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
        auc = func3(TARGETS,PROBS,target_index)
        auc_20 = func4(TARGETS,PROBS,target_index,is_ext)
        return val_loss, acc, auc, auc_20

def get_oof(df_train):
  PROBS = []
  oof = []
  for fold in range(kfold):
    df_valid = df_train[df_train["val_fold"] == fold]
    val_dataset = Albu_Dataset(df=df_valid, 
                                phase="train", 
                                transforms=Albu_Transform(image_size=image_size),
                                aug=val_aug,
                                use_meta=use_meta)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model_path_fold_1 = model_path + "_fold{}_1.pth".format(fold)

    n_meta_features =  val_dataset.n_meta_features #無理やり組み込んだ
    target_index = val_dataset.target_index #無理やり組み込んだ
    model = Ef_Net(n_meta_features=n_meta_features, out_features=out_features)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path_fold_1))
    model.eval()
    this_PROBS = val_epoch(model, val_loader, is_ext=df_valid['is_ext'].values, n_test=n_val, get_output=True)
    PROBS.append(this_PROBS)
    oof.append(df_valid)
  oof = pd.concat(oof).reset_index(drop=True)
  func5(oof,PROBS,target_index)
  return oof


def get_fold(df_train):
  #ログのセット
  logger = setup_logger(LOG_DIR, LOG_NAME)
  config.log_config(logger)
  #初期化
  scores = []
  scores_20 = []
  PROBS = []
  dfs = []

  for fold in range(kfold):
    if fold not in fold_list:
      logger.info("Fold{0}はスキップされました".format(fold))
      continue
    logger.info("{0}Fold{1}{2}".format("="*20,fold,"="*20))
    model_path_fold_1 = model_path + "_fold{}_1.pth".format(fold)  # Path and filename to save model to
    model_path_fold_2 = model_path + "_fold{}_2.pth".format(fold)  # Path and filename to save model to
    best_val = 0  # Best validation score within this fold
    patience = es_patience  # Current patience counter
    val_auc_max = 0.
    val_auc_20_max = 0.

    #kfoldの番号ふり
    df_this = df_train[df_train["val_fold"] != fold]
    df_valid = df_train[df_train["val_fold"] == fold]
    #データセットインスタンスの生成
    train_dataset = Albu_Dataset(df=df_this, 
                            phase="train", 
                            transforms=Albu_Transform(image_size=image_size),
                            aug=train_aug,
                            use_meta=use_meta)
    val_dataset = Albu_Dataset(df=df_valid, 
                            phase="train", 
                            transforms=Albu_Transform(image_size=image_size),
                            aug=val_aug,
                            use_meta=use_meta)
    #データローダーにセッット
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    #モデルクラスの読み込み
    n_meta_features =  train_dataset.n_meta_features #無理やり組み込んだ
    target_index = train_dataset.target_index #無理やり組み込んだ
    model = Ef_Net(n_meta_features=n_meta_features, out_features=out_features)
    model = model.to(device)
    logger.info("device_GPU_True:{}".format(next(model.parameters()).is_cuda))

    #オプティマイザー、スケジューラ―のセット
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if USE_AMP:
      model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=1, verbose=True, factor=0.2)
    
   
    #エポック
    for epoch in range(epochs):
      logger.info("{0} Epoch:{1}".format(time.ctime(),  epoch))
      train_loss = train_epoch(model, train_loader, optimizer,logger)
      val_loss, acc, val_auc, val_auc_20 = val_epoch(model, val_loader, is_ext=df_valid['is_ext'].values, target_index=target_index)
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
          logger.info("save_path={}".format(model_path_fold_1))
          val_auc_max = val_auc
          patience = es_patience  # Resetting patience since we have new best validation accuracy
          torch.save(model.state_dict(), model_path_fold_1)  # Saving current best model
      if val_auc_20 > val_auc_20_max:
          logger.info('val_auc_20_max ({:.6f} --> {:.6f}). Saving model ...'.format(val_auc_20_max, val_auc_20))
          logger.info("save_path={}".format(model_path_fold_1))
          torch.save(model.state_dict(), model_path_fold_2)
          val_auc_20_max = val_auc_20
  
    scores.append(val_auc_max)
    scores_20.append(val_auc_20_max)
  logger.info("train_finish")

def oof_tocsv(oof):
  oof.to_csv(oof_path, index=False)

def run(df_train):
    warnings.simplefilter('ignore')
    seed_everything(1)
    df_train = get_val_fold(df_train)
    get_fold(df_train)
    oof = get_oof(df_train)
    oof_tocsv(oof)
    return oof

#実行関数
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