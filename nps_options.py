#初期値にはデフォルト値を入力、Noneは全部表示
#numpy,pandas,snsそれぞれのオプション
# https://note.nkmk.me/python-numpy-set-printoptions-float-formatter/
# https://note.nkmk.me/python-pandas-option-display/
# https://note.nkmk.me/python-matplotlib-seaborn-basic/

import numpy as np
import pandas as pd
import seaborn as sns
print("インポート成功、関数一覧↓")
print("nps_options()")
def nps_options(thre=1000,row = 60,col = 20, style="darkgrid"):
    #np,pdの表示オプションの変更
    np.set_printoptions(threshold=thre)
    pd.options.display.max_rows = row
    pd.options.display.max_columns = col

    print("numpy一覧",np.get_printoptions())
    print("表示行数",pd.options.display.max_rows)
    print("表示列数",pd.options.display.max_columns)

    #スタイル
    if style == None:
        print("sスタイル:デフォルト")
        sns.set()
    else:
        print("sスタイル:",style)
        sns.set_style(style)