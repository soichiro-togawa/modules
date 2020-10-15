#自作モジュール
import numpy as np
from sklearn.metrics import roc_auc_score

def branch(criterion):
  if str(criterion) == "BCELoss()":
    def func1(probs,l):
      probs += l
      return probs
    def func2(logits,target):
      loss = criterion(logits, target.unsqueeze(1))
      return loss
    def func3(TARGETS,PROBS,mel_idx):
      auc = roc_auc_score(TARGETS.astype(float), PROBS)
      return auc
    def func4(TARGETS,PROBS,mel_idx,is_ext):
      auc_20 = roc_auc_score((TARGETS[is_ext==0]).astype(float), PROBS[is_ext==0])
      return auc_20
    def func5(oof,PROBS,mel_idx):
      oof['pred'] = np.concatenate(PROBS).squeeze()
      return oof
    def func6(OUTPUTS,PROBS,mel_idx):
      OUTPUTS.append(PROBS)

  elif str(criterion) == "BCEWithLogitsLoss()":
    def func1(probs,l):
      probs += l.sigmoid()
      return probs
    def func2(logits,target):
      loss = criterion(logits, target.unsqueeze(1))
      return loss
    def func3(TARGETS,PROBS,mel_idx):
      auc = roc_auc_score(TARGETS.astype(float), PROBS)
      return auc
    def func4(TARGETS,PROBS,mel_idx):
      auc_20 = roc_auc_score((TARGETS[is_ext==0]).astype(float), PROBS[is_ext==0])
      return auc_20
    def func5(oof,PROBS,mel_idx):
      oof['pred'] = np.concatenate(PROBS).squeeze()
      return oof
    def func6(OUTPUTS,PROBS,mel_idx):
      OUTPUTS.append(PROBS)

  elif str(criterion) == "CrossEntropyLoss()":
    def func1(probs,l):
      probs += l.softmax(1)
      return probs
    def func2(logits,target):
      loss = criterion(logits, target)
      return loss
    def func3(TARGETS,PROBS,mel_idx):
      auc = roc_auc_score((TARGETS==mel_idx).astype(float), PROBS[:, mel_idx])
      return auc
    def func4(TARGETS,PROBS,mel_idx,is_ext):
      auc_20 = roc_auc_score((TARGETS[is_ext==0]==mel_idx).astype(float), PROBS[is_ext==0, mel_idx])
      return auc_20
    def func5(oof,PROBS,mel_idx):
      oof['pred'] = np.concatenate(PROBS).squeeze()[:, mel_idx]
      return oof
    def func6(OUTPUTS,PROBS,mel_idx):
      OUTPUTS.append(PROBS[:, mel_idx])
  return func1,func2,func3,func4,func5,func6