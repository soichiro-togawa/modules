#!/usr/bin/bash
source "/mnt/d/Google ドライブ/00Colab Notebooks/00Commands/03SCRIPTkey/github.txt"
repoURL="https://github.com/soichiro-togawa/Git_private.git"
#コマンドライン引数で上書き可能
if [ $1 ]; then
  repoURL=$1
fi

git init
git config --global user.name soichiro-togawa
git config --global user.email soichiro.togawa@gmail.com
git remote add origin $repoURL
git remote -v