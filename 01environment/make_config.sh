#!/usr/bin/bash
config_path="config.txt"
host="Host my_ssh"
read -p "パブリックDNS:" host_name
# host_name="54.64.22.197"
user_name="uzu_kumaru"
key_path="~/.ssh/kagi"

touch $config_path
echo $host > $config_path
echo "  Hostname "$host_name >> $config_path
echo "  Port 22" >> $config_path
echo "  User "$user_name >> $config_path
echo "  IdentityFile "$key_path >> $config_path

# source "/mnt/d/Google ドライブ/Pipeline/01environment/make_config.sh"