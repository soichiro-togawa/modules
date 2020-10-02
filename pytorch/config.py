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
train_aug,val_aug,test_aug = True,False,True
target = "target"
TTA = 12 # Test Time Augmentation rounds
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
  TTA = 2
  #path系
  temp_model_path = model_path
  temp_oof_path = oof_path
  temp_predict_path = predict_path
  model_path =  model_path + "DEBUG_"
  oof_path =  oof_path + "DEBUG_"
  predict_path =  predict_path + "DEBUG_"
  LOG_NAME = "DEBUG_" + LOG_NAME

