#!/usr/bin/bash
source "/mnt/d/Google ドライブ/00Colab Notebooks/00Commands/03SCRIPTkey/aws.txt"
user_name="ubuntu"

echo `pwd`
chmod 700 "${key_path}"
sudo ssh -i "${key_path}" "$user_name@$1"

# source "/mnt/d/Google ドライブ/Pipeline/01environment/aws_ssh.sh" "ec2-54-226-50-27.compute-1.amazonaws.com"