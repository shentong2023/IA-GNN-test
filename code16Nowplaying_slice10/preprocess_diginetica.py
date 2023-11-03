from datetime import timedelta
import pandas as pd
import numpy as np
import pickle

# dataset url : https://competitions.codalab.org/competitions/11161
# data config
DATA_PATH = '../datasets/diginetica/raw/'
DATA_PATH_PREPROCESSED = '../datasets/diginetica/preprocessed/'
DATA_FILE = 'train-item-views.csv'

# filtering config
MIN_SESSION_LENGTH = 2
MIN_ITEM_SUPPORT = 5

# days test config
DAYS_TEST = 7


def preprocess(path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PREPROCESSED, min_item_support=MIN_ITEM_SUPPORT,
               min_session_length=MIN_SESSION_LENGTH, days_test=DAYS_TEST):
    data = load_data(path + file)
    data = filter_data(data, min_item_support, min_session_length)
    train, test = split_data(data, days_test)
    train_seqs, test_seqs = trans_data(train, test)
    write_data(train_seqs, path_proc, 'train.txt')
    write_data(test_seqs, path_proc, 'test.txt')


def load_data(file):
    data = pd.read_csv(file, sep=';', header=0, usecols=[0, 2, 3, 4],
                       dtype={0: np.int32, 2: np.int32, 3: np.int32, 4: str})
    data.columns = ['session_id', 'item_id', 'time', 'date']
    data['date_tmp'] = pd.to_datetime(data.date, format='%Y-%m-%d')

    print('loaded dataset\n\tevents:{}\n\tsessions:{}\n\taverage session length:{}\n\titems:{}\n\tspan:{} / {}\n\n'.format(
        len(data), data.session_id.nunique(), data.groupby('session_id').size().mean(), data.item_id.nunique(), data.date_tmp.min(), data.date_tmp.max()
    ))
    return data


def filter_data(data, min_item_support, min_session_length):
    # filter session length
    session_length = data.groupby('session_id').size()
    data = data[np.in1d(data.session_id, session_length[session_length >= min_session_length].index)]

    # filter item support
    item_support = data.groupby('item_id').size()
    data = data[np.in1d(data.item_id, item_support[item_support >= min_item_support].index)]

    # filter session length again
    session_length = data.groupby('session_id').size()
    data = data[np.in1d(data.session_id, session_length[session_length >= min_session_length].index)]

    print('filtered dataset\n\tevents:{}\n\tsessions:{}\n\titems:{}\n\tspan:{} / {}\n\n'.format(
        len(data), data.session_id.nunique(), data.item_id.nunique(), data.date_tmp.min(), data.date_tmp.max()
    ))
    return data


def split_data(data, days_test):
    session_max_times = data.groupby('session_id').date_tmp.max()
    split_time = data.date_tmp.max() - timedelta(days=days_test)

    # split data
    train = data[np.in1d(data.session_id, session_max_times[session_max_times < split_time].index)]
    test = data[np.in1d(data.session_id, session_max_times[session_max_times >= split_time].index)]

    # make sure items in test set have appeared in train set
    test = test[np.in1d(test.item_id, train.item_id)]
    session_length = test.groupby('session_id').size()
    test = test[np.in1d(test.session_id, session_length[session_length >= 2].index)]

    print('split dataset\n\ttrain set events:{}\n\tsessions:{}\n\titems:{}\n\tspan:{} / {}\n\n'.format(
        len(train), train.session_id.nunique(), train.item_id.nunique(), train.date_tmp.min(), train.date_tmp.max()
    ))
    print('split dataset\n\ttest set events:{}\n\tsessions:{}\n\titems:{}\n\tspan:{} / {}\n\n'.format(
        len(test), test.session_id.nunique(), test.item_id.nunique(), test.date_tmp.min(), test.date_tmp.max()
    ))
    print('average length:{:.2f}\n\n'.format((len(train)+len(test)) / (train.session_id.nunique()+test.session_id.nunique())))

    return train, test


def trans_data(train, test):
    """renumber items to start from 1 and generate sequences"""
    train.sort_values(['session_id', 'time'], ascending=True, inplace=True)
    test.sort_values(['session_id', 'time'], ascending=True, inplace=True)

    count = 1
    item_dict = {}  # key: previous item_id ; value: new item_id

    train_seqs = []  # e.g.[[1,2,3,4],[5,3,2],[6,7],...]
    cur_train_seq = []
    cur_session_id = -1
    for row in train.itertuples():
        item_id = getattr(row, 'item_id')
        if item_id not in item_dict:
            item_dict[item_id] = count
            count += 1

        session_id = getattr(row, 'session_id')
        if cur_train_seq and session_id != cur_session_id:
            train_seqs += [cur_train_seq]
            cur_train_seq = [item_dict[item_id]]
            cur_session_id = session_id
        else:
            cur_train_seq += [item_dict[item_id]]
    train_seqs += [cur_train_seq]  # add the last one

    test_seqs = []
    cur_test_seq = []
    cur_session_id = -1
    for row in test.itertuples():
        item_id = getattr(row, 'item_id')
        session_id = getattr(row, 'session_id')
        if cur_test_seq and session_id != cur_session_id:
            test_seqs += [cur_test_seq]
            cur_test_seq = [item_dict[item_id]]
            cur_session_id = session_id
        else:
            cur_test_seq += [item_dict[item_id]]
    test_seqs += [cur_test_seq]  # add the last one

    print('trans dataset\n\ttrain set sessions:{}\n\ttest set sessions:{}\n\n'.format(
        len(train_seqs), len(test_seqs)
    ))

    return train_seqs, test_seqs


def write_data(seqs, path_proc, file):
    inputs = []  # e.g.[[23],[23,6],[23,6,12],...]
    targets = []  # e.g.[[6],[12],[34],...]
    for seq in seqs:
        for i in range(1, len(seq) - 1):
            inputs += [seq[:i]]
            targets += [seq[i]]
    pickle.dump((inputs, targets), open(path_proc + file, 'wb'))

    print('write dataset\n\t ' + file + '--sessions after processing:{}\n\n'.format(
        len(inputs)
    ))


if __name__ == '__main__':
    preprocess()
