#自作モジュール
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

def branch(criterion):
  if str(criterion) == "BCELoss()":
    def func0(df_train,target):
      bi_target = {
          0:0, 1:0, 2:0,
          3:0, 4:0, 5:0,
          6:1, 7:0, 8:0}
      df_train[target] = df_train[target].map(bi_target)
      print(df_train[target].value_counts())
      return df_train
    def func1(probs,l):
      probs += l
      return probs
    def func2(logits,target):
      loss = criterion(logits, target.unsqueeze(1))
      return loss
    def func3(TARGETS,PROBS,target_index):
      auc = roc_auc_score(TARGETS.astype(float), PROBS)
      return auc
    def func4(TARGETS,PROBS,target_index,is_ext):
      auc_20 = roc_auc_score((TARGETS[is_ext==0]).astype(float), PROBS[is_ext==0])
      return auc_20
    def func5(oof,PROBS,target_index):
      oof['pred'] = np.concatenate(PROBS).squeeze()
      return oof
    def func6(OUTPUTS,PROBS,target_index):
      OUTPUTS.append(PROBS)

  elif str(criterion) == "BCEWithLogitsLoss()":
    def func0(df_train,target):
      bi_target = {
          0:0, 1:0, 2:0,
          3:0, 4:0, 5:0,
          6:1, 7:0, 8:0}
      df_train[target] = df_train[target].map(bi_target)
      print(df_train[target].value_counts())
      return df_train    
    def func1(probs,l):
      probs += l.sigmoid()
      return probs
    def func2(logits,target):
      loss = criterion(logits, target.unsqueeze(1))
      return loss
    def func3(TARGETS,PROBS,target_index):
      auc = roc_auc_score(TARGETS.astype(float), PROBS)
      return auc
    def func4(TARGETS,PROBS,target_index):
      auc_20 = roc_auc_score((TARGETS[is_ext==0]).astype(float), PROBS[is_ext==0])
      return auc_20
    def func5(oof,PROBS,target_index):
      oof['pred'] = np.concatenate(PROBS).squeeze()
      return oof
    def func6(OUTPUTS,PROBS,target_index):
      OUTPUTS.append(PROBS)

  elif str(criterion) == "CrossEntropyLoss()":
    def func0(df_train,target):
      return df_train  
    def func1(probs,l):
      probs += l.softmax(1)
      return probs
    def func2(logits,target):
      loss = criterion(logits, target)
      return loss
    def func3(TARGETS,PROBS,target_index):
      auc = roc_auc_score((TARGETS==target_index).astype(float), PROBS[:, target_index])
      return auc
    def func4(TARGETS,PROBS,target_index,is_ext):
      auc_20 = roc_auc_score((TARGETS[is_ext==0]==target_index).astype(float), PROBS[is_ext==0, target_index])
      return auc_20
    def func5(oof,PROBS,target_index):
      oof['pred'] = np.concatenate(PROBS).squeeze()[:, target_index]
      return oof
    def func6(OUTPUTS,PROBS,target_index):
      OUTPUTS.append(PROBS[:, target_index])
  return func0,func1,func2,func3,func4,func5,func6