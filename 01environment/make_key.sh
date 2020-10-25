#!/usr/bin/bash
#クオーテーションで囲まない
key_path="/mnt/c/Users/odenn/kagi"
read -p "key_pass:" pw

expect -c "
set timeout 5
#expectするコマンド
spawn ssh-keygen -t rsa -b 4096 -f $key_path
expect \"Enter passphrase (empty for no passphrase)\"
send \"$pw\n\"
expect \"Enter same passphrase again\"
send \"$pw\n\"
interact
"
echo `pwd`
temp=`cat "$key_path.pub"`
echo $temp

# source "/mnt/d/Google ドライブ/Pipeline/01environment/make_key.sh"