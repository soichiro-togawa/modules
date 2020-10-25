#!/usr/bin/bash
ab="a"

expect -c "
set timeout 5
#expectするコマンド
spawn ssh-keygen -t rsa -b 4096 -f "/mnt/c/Users/odenn/kagi"
expect \"Enter passphrase (empty for no passphrase):\"
send \"$ab\n\"
expect \"Enter same passphrase again:\"
send \"$ab\n\"
interact
"

# echo `pwd`
# cat "kagi.pub"
# rm "/mnt/c/Users/odenn/kagi"
# rm "/mnt/c/Users/odenn/kagi.pub"

# source "/mnt/d/Google ドライブ/Pipeline/01environment/make_key.sh"