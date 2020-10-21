#configは辞書で持たせる！！！！！！！！！！！！！！！！！！！！！！！！！
#config→starter.sh(空白を入れないこと！！！！！)とpreprocess.py用
data_dir1="/content/image/"
data_dir2="/content/image_2019/"

#preprocess.py
import torch, torch.nn as nn
PREPROCESS_CONFIG ={
"image_name": "image_name",
"target": "target",
"extension": ".jpg",
"use_meta": True,
"use_external": True,
"criterion": nn.BCEWithLogitsLoss(),
"criterion": nn.CrossEntropyLoss(),
}

#model.py
MODEL_CONFIG ={
"b_num": "b1",
"out_features": 1,
"out_features": 9,
"criterion": PREPROCESS_CONFIG["criterion"],
"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
#既出
"use_meta": PREPROCESS_CONFIG["use_meta"],
}

#train.pyとpredict.py
TRAIN_CONFIG ={
"DEBUG": True,
"USE_AMP": False,
"image_size": 256,
"epochs": 12,
"es_patience": 3,  # Early Stopping patience
"batch_size": 64,
"num_workers": 8,
"kfold": 5,
"fold_list":[0,1,2,3,4], #実行したいfold番号をリストに(0始まり)
# "fold_list":[2,3,4],
"train_aug": True, "val_aug": False, "test_aug": False,
"n_val": 8, "TTA": 8,  #n_val→get_transの回数
#既出
"use_meta": PREPROCESS_CONFIG["use_meta"],"target": PREPROCESS_CONFIG["target"],
"out_features": MODEL_CONFIG["out_features"],"criterion": MODEL_CONFIG["criterion"],"device": MODEL_CONFIG["device"],
}

#path系
PATH_CONFIG ={
"df_train_preprocessed_path": "/content/df_train.csv",
"df_test_preprocessed_path": "/content/df_test.csv",
"VERSION": 2,
"model_path": "/content/drive/My Drive/00Colab Notebooks/11Kaggle/melanoma/model/",
"oof_path": "/content/drive/My Drive/00Colab Notebooks/11Kaggle/melanoma/sub_3/",
"predict_path": "/content/drive/My Drive/00Colab Notebooks/11Kaggle/melanoma/sub_3/",
#log
"LOG_DIR": "/content/drive/My Drive/Pipeline/_output_dir/",
}
PATH_CONFIG["model_name"] ="ef{0}_im{1}_amp{2}_ver{3}".format(MODEL_CONFIG["b_num"],TRAIN_CONFIG["image_size"],TRAIN_CONFIG["USE_AMP"],PATH_CONFIG["VERSION"])
PATH_CONFIG["LOG_NAME"] ="training_log_" + PATH_CONFIG["model_name"]


#DEBUGモードとpathの設定
temp_model_path = PATH_CONFIG["model_path"]
temp_oof_path = PATH_CONFIG["oof_path"]
temp_predict_path = PATH_CONFIG["predict_path"]
if TRAIN_CONFIG["DEBUG"] == True:
    TRAIN_CONFIG["epochs"] = 1
    TRAIN_CONFIG["kfold"] = 2
    TRAIN_CONFIG["n_val"],TRAIN_CONFIG["TTA"] =1, 2
    #path系
    PATH_CONFIG["model_name"] = "DEBUG_" + PATH_CONFIG["model_name"]
    PATH_CONFIG["LOG_NAME"] = "DEBUG_" + PATH_CONFIG["LOG_NAME"]
PATH_CONFIG["model_path"] =  PATH_CONFIG["model_path"] + PATH_CONFIG["model_name"] #foldと.pthはtrain.pyで足す
PATH_CONFIG["oof_path"] =  PATH_CONFIG["oof_path"] + "OOF_" + PATH_CONFIG["model_name"] + ".csv"
PATH_CONFIG["predict_path"] =  PATH_CONFIG["predict_path"] + "PREDICT_" + PATH_CONFIG["model_name"] + ".csv"


#コンフィグのログ
def log_config(logger):
    import os
    if TRAIN_CONFIG["DEBUG"]==True:
      logger.info("#########DEBUG-MODE#############")
    logger.info("#####Print_Config######")
    logger.info(PREPROCESS_CONFIG)
    logger.info(MODEL_CONFIG)
    logger.info(TRAIN_CONFIG)
    logger.info("----------------path------------------")
    logger.info(PATH_CONFIG)
    logger.info("model_path_exist:{}".format(os.path.exists(temp_model_path)))
    logger.info("oof_path_exist:{}".format(os.path.exists(temp_oof_path)))
    logger.info("predict_path_exist:{}".format(os.path.exists(temp_predict_path)))
    logger.info("LOG_DIR_exist:{}".format(os.path.exists(PATH_CONFIG["LOG_DIR"])))
    logger.info("#####END######")