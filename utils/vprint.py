#変数名:valueを表示→ローカル変数にも対応
# https://qiita.com/AnchorBlues/items/f7725ba87ce349cb0382

from inspect import currentframe
def vprint(*args):
    names = {id(v):k for k,v in currentframe().f_back.f_locals.items()}
    print(', '.join(names.get(id(arg),'???')+' := '+repr(arg) for arg in args))




#メモ
# pip install varname
# from varname import nameof
# aa = "apple"
# nameof(aa)
##############使用方法###############
# from utils.vprint import vprint