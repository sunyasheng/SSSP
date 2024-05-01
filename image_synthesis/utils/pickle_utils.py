import pickle


def read_pickle(fp):
    with open(fp, 'rb') as f:
        data = pickle.load(f)
    return data
