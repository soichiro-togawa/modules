FROM ubuntu:latest
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

RUN pip install --upgrade pip
WORKDIR /
#0.0.0.0はlocalhostの意味
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--LabApp.token=''"]


#######使用方法########
# docker build .
# docker images
# docker run -p 8888:8888 -v <mnt>:/work --name <name> <imageid>
# ブラウザでlocalhost:8888, <name>はなんでもよい , workフォルダはubuntu側に自動生成