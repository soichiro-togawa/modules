FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
RUN apt-get update && apt-get install -y\
    sudo\
    wget\
    vim
#/optに置いた方がroot以外のユーザーが触りやすい
WORKDIR /opt
#copyだと現物ファイルをとってくる手間があるので、wgetでRUNする
RUN wget https://repo.continuum.io/archive/Anaconda3-2019.10-Linux-x86_64.sh &&\
    sh /opt/Anaconda3-2019.10-Linux-x86_64.sh -b -p /opt/anaconda3 &&\
    rm -f Anaconda3-2019.10-Linux-x86_64.sh
ENV PATH /opt/anaconda3/bin:$PATH

RUN pip install --upgrade pip && pip install\
	keras==2.3\
	scipy==1.4.1\
	tensorflow-gpu==2.1
WORKDIR /
#0.0.0.0はlocalhostの意味
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--LabApp.token=''"]

##############使用方法##############
# docker build .
# docker images
# docker run --gpus all -v <mnt_aws>:/work -p 8888:8888 <imageid>
# ブラウザでPublicDNS:8888, --gpus allで使用するgpuの個数を指定, workフォルダはubuntu側に自動生成
# ターミナルからnvidia-smiでGPUの使用を確認
