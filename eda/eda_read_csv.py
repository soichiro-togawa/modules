import pandas as pd

def read_csv(dir, train_path="train.csv", test_path="test_csv", sub_path=None):
    df_train = pd.read_csv(dir + "/" + train_path)
    if test_path:
        df_test = pd.read_csv(dir + "/" + test_path)
        if sub_path:
            df_sub = pd.read_csv(dir + "/" + sub_path)
            return df_train, df_test, df_sub
        else:
            return df_train, df_test
    elif sub_path:
        df_sub = pd.read_csv(dir + "/" + sub_path)
        return df_train, df_sub
    else:
        return df_train

#trainデータカラムと、testデータカラムの相違（ターゲットの特定）
def search_targets(df_train, df_test):
    targets = list(set(df_train.columns) ^ set(df_test.columns))
    print(targets)
    if len(targets):
        target = targets[0]
    else:
        target = targets
    return target


def target_plot():
    sns.distplot(df_train[target], fit=norm)
    fig = plt.figure(figsize(10,10))
    res = stats.probplot(df_train[target],plot=plt)