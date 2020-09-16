import pandas as pd
import numpy as np

#アンサンブルモジュール
class EnsembleSub():
    def __init__(self, ensemble_list, targets = ["target"]):
        self.ensemble_list = ensemble_list
        self.targets = targets
        self.len = len(ensemble_list)
        for i in range(self.len):
          self.ensemble_list[i] = pd.read_csv(self.ensemble_list[i])
        self.result = self.ensemble_list[0].copy()
        for i in self.targets:
          self.result[i] = 0

    #単純平均
    def add(self):
      result = self.result.copy()
      for i in self.targets:
        for j in range(len(self.ensemble_list)):
          result[i] += self.ensemble_list[j][i]
        result[i] /=  self.len   
      return result

    #係数がけ平均
    def add_cof(self, cof_list):
      result = self.result.copy()
      for i in self.targets:
        for j in range(len(self.ensemble_list)):
          result[i] += (self.ensemble_list[j][i] * cof_list[j])
        result[i] /=  self.len   
      return result

    #min_max(閾値を満たさない値はベースを参照)
    def min_max(self, high=0.9, low=0.1, base_index = 0):
      result = self.result.copy()
      for i in self.targets:
        result[i] = (self.ensemble_list[base_index][i])

      for i in self.targets:
        df = pd.concat([j[i] for j in self.ensemble_list],axis=1)
        high_idx = df[df >= high].dropna(how='any').index
        low_idx = df[df <= low].dropna(how='any').index
        for k in high_idx:
          result[i][k] = df.max(axis=1)[k]
        for k in low_idx:
          result[i][k] = df.min(axis=1)[k]
      return result


#########使用方法##########
# from ensemble import EnsembleSub
# path_list = ["/content/drive/My Drive/00Colab Notebooks/11Kaggle/melanoma/sub/submission_05.csv",
#              "/content/drive/My Drive/00Colab Notebooks/11Kaggle/melanoma/sub/emsamble.csv",
#              "/content/drive/My Drive/00Colab Notebooks/11Kaggle/melanoma/sub/submission_05.csv"
#              ]

# instance = EnsembleSub(path_list)
# re1 = instance.add()
# re2 = instance.add_cof([0.5, 0.25, 0.25])
# #path_listの順番がインデックスの順番→base_index=0ではpath_list[0]が指定されている
# re3 = instance.min_max(high=0.9, low=0.1, base_index=0)