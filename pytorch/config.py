#config→starter.sh(空白を入れないこと！！！！！)とpreprocess.py用
image_dir1="/content/image/"
image_dir2="/content/image_2019/"

#Config train.pyとpredict.py用
DEBUG = True
USE_AMP = False
image_size = 256
epochs = 12  # Number of epochs to run
es_patience = 3  # Early Stopping patience - for how many epochs with no improvements to wait
batch_size = 64
num_workers = 8
kfold = 5
b_num = "b1"
out_features = 1
import torch.nn as nn,torch
criterion = nn.BCEWithLogitsLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_meta = False
train_aug,val_aug,test_aug = True,False,False
target = "target"
n_val,TTA = 8,12 # Test Time Augmentation rounds
VERSION = 1
#path系
model_name ="ef{0}_im{1}_amp{2}_ver{3}".format(b_num, image_size, USE_AMP, VERSION)
model_path = "/content/drive/My Drive/00Colab Notebooks/11Kaggle/melanoma/model/"
oof_path = "/content/drive/My Drive/00Colab Notebooks/11Kaggle/melanoma/sub_3/"
predict_path = "/content/drive/My Drive/00Colab Notebooks/11Kaggle/melanoma/sub_3/"
#log
LOG_DIR = "/content/drive/My Drive/Pipeline/output_dir/"
LOG_NAME = "training_log_" + model_name

if DEBUG == True:
  epochs = 1
  kfold = 2
  n_val,TTA =1, 2
  #path系
  temp_model_path = model_path
  temp_oof_path = oof_path
  temp_predict_path = predict_path
  model_path =  model_path + "DEBUG_"
  oof_path =  oof_path + "DEBUG_"
  predict_path =  predict_path + "DEBUG_"
  LOG_NAME = "DEBUG_" + LOG_NAME


#コンフィグのログ
def log_config(logger):
    import os
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
      logger.info("model_path:{}".format(os.path.exists(temp_model_path)))
      logger.info("oof_path:{}".format(os.path.exists(temp_oof_path)))
      logger.info("predict_path:{}".format(os.path.exists(temp_predict_path)))
    logger.info("#####END######")
#コンフィグプリント
def print_config():
    import os
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
