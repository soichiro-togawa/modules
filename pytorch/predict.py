#トレーニング＋predict oof
from pytorch.dataset import Albu_Dataset
from pytorch.transform import Albu_Transform
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import warnings,os


#コンフィグの読み直し！！
from pytorch import config
import pytorch
import importlib
importlib.reload(pytorch.config)
#predict_config
DEBUG = config.DEBUG
image_size = config.image_size
batch_size = config.batch_size
num_workers = config.num_workers
kfold = config.kfold
test_aug = config.test_aug
target = config.target
TTA = config.TTA
model_name = config.model_name
model_path = config.model_path
predict_path = config.predict_path
oof_path = config.oof_path

def print_config():
    print("DEBUG",DEBUG)
    print("image_size",image_size)
    print("batch_size",batch_size)
    print("num_workers",num_workers)
    print("kfold",kfold)
    print("target",target)
    print("test_aug",test_aug)
    print("TTA",TTA)
    print("--------path----------")
    print("model_name",model_name)
    print("model_path",model_path,os.path.exists(model_path))
    print("predict_path",predict_path,os.path.exists(predict_path))
    print("oof_path",oof_path,os.path.exists(oof_path))
    if DEBUG==True:
      print("#########DEBUG-MODE-predict#############")
      print("model_path",os.path.exists(config.temp_model_path))
      print("oof_path",os.path.exists(config.temp_oof_path))
      print("predict_path",os.path.exists(config.temp_predict_path))
print_config()


def main(df_test, imfolder_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device_CPU_GPU:", device)
    predict = torch.zeros((len(df_test), 1), dtype=torch.float32, device=device) 
    test_dataset = Albu_Dataset(df=df_test,
                imfolder=imfolder_test,
                phase="test",
                transforms=Albu_Transform(image_size=image_size),
                aug=test_aug)
    for fold in range(kfold):
        print('=' * 20, 'Fold', fold, '=' * 20)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        model_path_fold = model_path + model_name + "_fold{}.pth".format(fold)
        model = torch.load(model_path_fold)
        model.eval()
        with torch.no_grad():
          tta_predict = torch.zeros((len(test_dataset), 1), dtype=torch.float32, device=device)
          for _ in range(TTA):
              for i, x_test in enumerate(test_loader):
                  x_test = torch.tensor(x_test, device=device, dtype=torch.float32)
                  output_test = model(x_test)
                  output_test = torch.sigmoid(output_test)
                  tta_predict[i*test_loader.batch_size:i*test_loader.batch_size + x_test.shape[0]] += output_test
          predict += tta_predict / TTA
    predict /= kfold
    return predict
    
def predict_tocsv(df_sub, predict_temp):
    df_sub[target] = predict_temp.cpu().numpy().reshape(-1,)
    df_sub.to_csv(predict_path + "predict_{}.csv".format(model_name), index=False)

def run(df_test,df_sub,imfolder_test):
    from pytorch.seed import seed_everything
    seed_everything(1)
    warnings.simplefilter('ignore')
    test_predict = main(df_test, imfolder_test)
    predict_tocsv(df_sub, test_predict)
    return test_predict

if __name__ == '__main__':
    run(df_test,df_sub,imfolder_test)

#########使用方法##########
# import importlib
# from pytorch import predict
# import pytorch.predict
# importlib.reload(pytorch.predict)
# oof = predict.run(df_test,df_sub,imfolder_test)