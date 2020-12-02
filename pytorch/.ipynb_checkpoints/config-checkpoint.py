#For training
DEBUG = True
image_size = 256
epochs = 12  # Number of epochs to run
es_patience = 3  # Early Stopping patience - for how many epochs with no improvements to wait
batch_size = 32
num_workers = 8
TTA = 12 # Test Time Augmentation rounds
kfold = 5
save_path = "/content/drive/My Drive/00Colab Notebooks/11Kaggle/melanoma/model/"
b_num = "b1"
train_aug,val_aug,test_aug = False,False,False
