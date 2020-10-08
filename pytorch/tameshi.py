import sys,os
print("f",__file__)
print("p",__package__)
print("n",__name__)
# print(sys.modules)
# print(len(sys.modules))
print(os.getcwd())
# print(sys.path)
# sys.path.append("/content/drive/My Drive/Pipeline")
for i,j in enumerate(sys.path):
  print(i,j)
print("len:",len(sys.path))
op = "opop"
# #sys.pathでのパスはpwd地点からのパス→import時のパスは、インポートに使用された環境変数からのパス
# sys.path.append(os.getcwd())
# print(sys.path)
# # sys.path.append("drive/My Drive/Pipeline")
# sys.path.append("/content/drive/My Drive/Pipeline")
# from ..utils import vprint
# # import config
# # from . import config
# # from .. import pipi
# # import pytorch
# from . import vprint
# import config
# from utils import vprint
# a = "abanana"
# print(a)
# vprint.vvprint(a)
# import pytorch
# import config
# import train
# import numpy as np
# np.app()
