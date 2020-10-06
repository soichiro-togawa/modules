import numpy as np, pandas as pd
import sys
sys.path.append("/content/drive/My Drive/Pipeline/utlis")
from vprint import vprint

def read_csv():
    dir_path = "/content/256/"
    #256
    imfolder_test = "/content/256/test/"
    imfolder_train = "/content/256/train/"
    imfolder_valid = "/content/256/train/"
    #256_2019
    imfolder_train2019 = "/content/256_2019/train/"
    df_train2019 = "/content/256_2019/train.csv"
    #512
    # imfolder_test = "/content/test/"
    # imfolder_train = "/content/train/"
    # imfolder_valid = "/content/train/"

    df_train = pd.read_csv(dir_path + "train.csv")
    df_test = pd.read_csv(dir_path + "test.csv")
    df_train2019 = pd.read_csv(df_train2019)

    #ファイルパスの記入
    df_train["image_path"] = imfolder_train + df_train["image_name"] + ".jpg"
    df_test["image_path"] = imfolder_test + df_test["image_name"] + ".jpg"
    df_train2019["image_path"] = imfolder_train2019 + df_train2019["image_name"] + ".jpg"
    return df_train,df_test,df_train2019

def get_preprocess(df_train,df_test,df_train2,binary=True):
    use_meta = True
    use_external = True

    #工程１
    df_train = df_train[df_train['tfrecord'] != -1].reset_index(drop=True)
    df_train['fold'] = df_train['tfrecord'] % 5
    #keyをvakueに変換
    tfrecord2fold = {
        2:0, 4:0, 5:0,
        1:1, 10:1, 13:1,
        0:2, 9:2, 12:2,
        3:3, 8:3, 11:3,
        6:4, 7:4, 14:4,}
    df_train['fold'] = df_train['tfrecord'].map(tfrecord2fold)
    df_train['is_ext'] = 0

    df_train['diagnosis'] = df_train['diagnosis'].apply(lambda x: x.replace('seborrheic keratosis', 'BKL'))
    df_train['diagnosis'] = df_train['diagnosis'].apply(lambda x: x.replace('lichenoid keratosis', 'BKL'))
    df_train['diagnosis'] = df_train['diagnosis'].apply(lambda x: x.replace('solar lentigo', 'BKL'))
    df_train['diagnosis'] = df_train['diagnosis'].apply(lambda x: x.replace('lentigo NOS', 'BKL'))
    df_train['diagnosis'] = df_train['diagnosis'].apply(lambda x: x.replace('cafe-au-lait macule', 'unknown'))
    df_train['diagnosis'] = df_train['diagnosis'].apply(lambda x: x.replace('atypical melanocytic proliferation', 'unknown'))
    df_train['diagnosis'] = df_train['diagnosis'].apply(lambda x: x.replace('NV', 'nevus'))
    df_train['diagnosis'] = df_train['diagnosis'].apply(lambda x: x.replace('MEL', 'melanoma'))

    print(df_train['diagnosis'].value_counts())

    #工程２
    if use_external:
        df_train2 = df_train2[df_train2['tfrecord'] >= 0].reset_index(drop=True)
        df_train2['fold'] = df_train2['tfrecord'] % 5
        df_train2['is_ext'] = 1
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('NV', 'nevus'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('MEL', 'melanoma'))
        df_train = pd.concat([df_train, df_train2]).reset_index(drop=True)

    #dianosis2のラベルエンコーディング(対応表)
    diagnosis2idx = {d: idx for idx, d in enumerate(sorted(df_train.diagnosis.unique()))}
    print(diagnosis2idx)
    df_train['target'] = df_train['diagnosis'].map(diagnosis2idx)
    mel_idx = diagnosis2idx['melanoma']
    vprint(mel_idx)

    #工程３
    from tqdm import tqdm
    import os
    if use_meta:
        # One-hot encoding of anatom_site_general_challenge feature
        concat = pd.concat([df_train['anatom_site_general_challenge'], df_test['anatom_site_general_challenge']], ignore_index=True)
        dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='site')
        df_train = pd.concat([df_train, dummies.iloc[:df_train.shape[0]]], axis=1)
        df_test = pd.concat([df_test, dummies.iloc[df_train.shape[0]:].reset_index(drop=True)], axis=1)
        # Sex features
        df_train['sex'] = df_train['sex'].map({'male': 1, 'female': 0})
        df_test['sex'] = df_test['sex'].map({'male': 1, 'female': 0})
        df_train['sex'] = df_train['sex'].fillna(-1)
        df_test['sex'] = df_test['sex'].fillna(-1)
        # Age features
        df_train['age_approx'] /= 90
        df_test['age_approx'] /= 90
        df_train['age_approx'] = df_train['age_approx'].fillna(0)
        df_test['age_approx'] = df_test['age_approx'].fillna(0)
        df_train['patient_id'] = df_train['patient_id'].fillna(0)
        # n_image per user
        df_train['n_images'] = df_train.patient_id.map(df_train.groupby(['patient_id']).image_path.count())
        df_test['n_images'] = df_test.patient_id.map(df_test.groupby(['patient_id']).image_path.count())
        df_train.loc[df_train['patient_id'] == -1, 'n_images'] = 1
        df_train['n_images'] = np.log1p(df_train['n_images'].values)
        df_test['n_images'] = np.log1p(df_test['n_images'].values)
        # image size
        train_images = df_train['image_path'].values
        train_sizes = np.zeros(train_images.shape[0])
        for i, img_path in enumerate(tqdm(train_images)):
            train_sizes[i] = os.path.getsize(img_path)
        df_train['image_size'] = np.log(train_sizes)
        test_images = df_test['image_path'].values
        test_sizes = np.zeros(test_images.shape[0])
        for i, img_path in enumerate(tqdm(test_images)):
            test_sizes[i] = os.path.getsize(img_path)
        df_test['image_size'] = np.log(test_sizes)
        meta_features = ['sex', 'age_approx', 'n_images', 'image_size'] + [col for col in df_train.columns if col.startswith('site_')]
        n_meta_features = len(meta_features)
    else:
        n_meta_features = 0
    print("\n",n_meta_features,meta_features)
    df_train.head(2)

    #二値モデル用
    if binary:
        bi_target = {
            0:0, 1:0, 2:0,
            3:0, 4:0, 5:0,
            6:1, 7:0, 8:0}
        df_train['target'] = df_train['target'].map(bi_target)
        print(df_train["target"].value_counts())
    return df_train,df_test