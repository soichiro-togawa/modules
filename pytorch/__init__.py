# import pytorchしたタイミングでパッケージ内全モジュールがリロードされる
# importlib.reload(pytorch)と組み合わせることで簡単にリロードできる
#全モジュールが読み込まれるから、おかしいコードが検出できる
import os,importlib
abs_path = os.path.dirname(os.path.abspath(__file__))
py_files = os.listdir(abs_path)
py_files.remove("__init__.py")
py_files = [s for s in py_files if ".py" in s]
py_files = [s.replace(".py", "") for s in py_files]
py_files = [__package__ +"."+s for s in py_files]
# print(py_files)
#モジュール同士が読みあってる場合、二回回さないと更新されない場合がある
for loop in range(2):
  for i in py_files:
    if i in ["pytorch.scheduler"]: #読み込みを飛ばしたファイルを記述
      continue
    a = importlib.import_module(i)
    # print(a.__name__)
    importlib.reload(a)

#使い方：クラス(Ef_Net)を読むときは必ず下に書く、モジュール(config)は上下どちらでもよい
# import pytorch
# importlib.reload(pytorch)
# from pytorch.model import Ef_Net
# from pytorch import config
