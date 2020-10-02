#!/usr/bin/bash
cd "/mnt/d/Google ドライブ/Pipeline"
source "/mnt/d/Google ドライブ/00Colab Notebooks/00Commands/03SCRIPTkey/github.txt"

STR="backup"
TIME="`date +%Y%m%d_%H%M%S`"
FILE="$STR"_"$TIME"

git checkout master
git branch "$FILE"
git checkout "$FILE"


expect -c "
set timeout 5
#expectするコマンド
spawn git push origin "$FILE"
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
# "/mnt/d/Google ドライブ/Pipeline/shell_script/git_make_backup.sh"