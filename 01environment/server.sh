#!/usr/bin/bash
read -p "pub_cat:" key_cat
mkdir ~/.ssh
chmod 700 ~/.ssh
cd ~/.ssh
echo $key_cat > authorized_keys
chmod 600 authorized_keys

#source server.sh