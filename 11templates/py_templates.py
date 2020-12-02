import argparse
import pandas as pd
import numpy as np
parser = argparse.ArgumentParser(description="説明")
# parser.add_argument('arg1', help='df_train')
# parser.add_argument('arg1', help='df_train',type = pd.DataFrame())
parser.add_argument('arg1', help='df_train',type = np.array)
args = parser.parse_args()
# print('arg1='+args.arg1)


if __name__ == '__main__':
  # print(args.arg1.head())
  import numpy as np
  print(args.arg1)
  print(args.arg1.shape)
