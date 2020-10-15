#!/bin/bash

#変数
image_size=256
apex=False
#config.pyから引用
source "/content/drive/My Drive/Pipeline/pytorch/config.py" >&/dev/null
dir1=${data_dir1}
dir2=${data_dir2}
unzip1="/content/drive/My Drive/00Colab Notebooks/11Kaggle/melanoma/data/256.zip"
unzip2="/content/drive/My Drive/00Colab Notebooks/11Kaggle/melanoma/data/256_2019.zip"
unzip3="/content/drive/My Drive/00Colab Notebooks/11Kaggle/melanoma/512.zip"
#コマンドライン引数で上書き可能
if [ $1 ]; then
  image_size=$1
fi
if [ $2 ]; then
  apex=$2
fi


echo "============開始============"
cd "/content"
mkdir $dir1
mkdir $dir2

if [ $image_size = 256 ]; then
  echo "============image_sizeは256================"
  unzip "$unzip1" -d "$dir1" 1>/dev/null
  unzip "$unzip2" -d "$dir2" 1>/dev/null
elif [ $image_size = 384 ]; then
  echo "================image_sizeは384================"
elif [ $image_size = 512 ]; then
  echo "================image_sizeは512================"
  unzip $unzip3 -d $dir1
else
  echo "================該当イメージサイズなし→unzip非実行================"
fi


echo "============pip関連============"
#pipインストールした後は、ランタイムを再起動した方がベター
pip install -r "/content/drive/My Drive/Pipeline/pytorch/requirement.txt" 1>&/dev/null

if [ $apex = True ]; then
  echo "============apexを使用します============"
  git clone "https://github.com/NVIDIA/apex"
  pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
else
  echo "============apexなし============"
fi


echo "============終了============"
###使用方法###
# !chmod 700 "/content/drive/My Drive/Pipeline/pytorch/starter.sh"
# !"/content/drive/My Drive/Pipeline/pytorch/starter.sh"