#!/usr/bin/bash
cd "/mnt/d/Google ドライブ/Pipeline"
source "/mnt/d/Google ドライブ/00Colab Notebooks/00Commands/03SCRIPTkey/github.txt"
git checkout master
git add -A
git commit -m "script_push"

expect -c "
set timeout 5
#expectするコマンド
spawn git push origin master
expect \"Username\"
send \"${UN}\n\"
expect \"Password\"
send \"${PW}\n\"
interact
#interactの変わり↓
# expect \"\\\$\"
# exit 0
"

#使い方
# "/mnt/d/Google ドライブ/Pipeline/shell_script/git_push.sh"