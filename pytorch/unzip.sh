#!/bin/bash
echo "開始"
mkdir "/content/256"
mkdir "/content/256_2019"
# unzip "/content/drive/My Drive/00Colab Notebooks/11Kaggle/melanoma/data/256.zip" -d "/content/256"
# unzip "/content/drive/My Drive/00Colab Notebooks/11Kaggle/melanoma/data/256_2019.zip" -d "/content/256_2019"
# unzip "/content/drive/My Drive/00Colab Notebooks/11Kaggle/melanoma/512.zip"

#pip関連
echo "requirement関連"
pip install -r "/content/drive/My Drive/Pipeline/pytorch/requirement.txt"

# echo "ディレクトリ移動"
# cd "/content/drive/My Drive/Pipeline"
# echo `pwd`
echo "終わったよ"


###使用方法###
# !chmod 700 "/content/drive/My Drive/Pipeline/pytorch/unzip.sh"
# !"/content/drive/My Drive/Pipeline/pytorch/unzip.sh"