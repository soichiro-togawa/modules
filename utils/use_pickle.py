import pickle

def save_pkl(path, data):
# with open("/content/json.pkl", mode='wb') as f:
with open(path, mode='wb') as f:
    pickle.dump(data,f)

def load_pkl(path)
with open(path, mode='rb') as f:
    return pickle.load(f)
