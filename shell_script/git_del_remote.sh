#!/usr/bin/bash
cd "/mnt/d/Google ドライブ/Pipeline"
source "/mnt/d/Google ドライブ/00Colab Notebooks/00Commands/03SCRIPTkey/github.txt"
git checkout master
#gitignoreにtxtついてないか注意
git branch -D $1


expect -c "
set timeout 5
#expectするコマンド
spawn git push --delete origin $1
expect \"Username\"
send \"${UN}\n\"
expect \"Password\"
send \"${PW}\n\"
interact
#interactの変わり↓
# expect \"\\\$\"
# exit 0
"
git checkout master

#使い方
# source "/mnt/d/Google ドライブ/Pipeline/shell_script/git_push_master.sh"