#Config train.pyとpredict.py用
DEBUG = True
image_size = 256
epochs = 12  # Number of epochs to run
es_patience = 3  # Early Stopping patience - for how many epochs with no improvements to wait
batch_size = 32
num_workers = 8
kfold = 5
b_num = "b1"
train_aug,val_aug,test_aug = True,False,True
target = "target"
TTA = 12 # Test Time Augmentation rounds
VERSION = 1
#path系
model_name ="ef{0}_im{1}_tta{2}_ver{3}".format(b_num, image_size, TTA, VERSION) 
model_path = "/content/drive/My Drive/00Colab Notebooks/11Kaggle/melanoma/model/"
oof_path = "/content/drive/My Drive/00Colab Notebooks/11Kaggle/melanoma/sub3/"
predict_path = "/content/drive/My Drive/00Colab Notebooks/11Kaggle/melanoma/sub3/"


if DEBUG == True:
  epochs = 2
  kfold = 2
  TTA = 2

  model_name ="ef{0}_im{1}_tta{2}_ver{3}".format(b_num, image_size, TTA, VERSION) 
  model_path =  model_path + "DEBUG_"
  oof_path =  model_path + "DEBUG_"
  predict_path =  model_path + "DEBUG_"

