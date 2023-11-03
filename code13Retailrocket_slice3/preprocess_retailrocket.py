from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pickle

# dataset url : https://www.kaggle.com/retailrocket/ecommerce-dataset?select=events.csv
# data config
DATA_PATH = '../datasets/retailrocket/raw/'
DATA_PATH_PREPROCESSED = '../datasets/retailrocket/preprocessed/'
DATA_FILE = 'events.csv'
SESSION_LENGTH = 30 * 60 * 1000  # 30 minutes -> milliseconds

# filtering config
MIN_SESSION_LENGTH = 2
MIN_ITEM_SUPPORT = 5

# days test config
DAYS_TEST = 7


def preprocess(path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PREPROCESSED, min_item_support=MIN_ITEM_SUPPORT,
               min_session_length=MIN_SESSION_LENGTH, days_test=DAYS_TEST, session_length=SESSION_LENGTH):
    data = load_data(path + file, session_length)
    data = filter_data(data, min_item_support, min_session_length)
    train, test = split_data(data, days_test)
    train_seqs, test_seqs = trans_data(train, test)
    write_data(train_seqs, path_proc, 'train.txt')
    write_data(test_seqs, path_proc, 'test.txt')


def load_data(file, session_length):
    data = pd.read_csv(file, sep=',', header=0, usecols=[0, 1, 2, 3],
                       dtype={0: np.int64, 1: np.int32, 2: str, 3: np.int32})
    data.columns = ['timestamp', 'user_id', 'action_type', 'item_id']

    data = data[data.action_type == 'view']  # only consider view action
    del data['action_type']

    data.sort_values(['user_id', 'timestamp'], ascending=True, inplace=True)
    data['timestamp_tmp'] = data['timestamp'].shift()  # move down one line
    data['time_delta'] = abs(data['timestamp'] - data['timestamp_tmp'])
    data['session_id'] = (data['time_delta'] > session_length).cumsum()

    del data['timestamp_tmp']
    del data['time_delta']

    data.sort_values(['session_id', 'timestamp'], ascending=True, inplace=True)

    print(
        'loaded dataset\n\tevents:{}\n\tsessions:{}\n\taverage session length:{}\n\titems:{}\n\tspan:{} / {}\n\n'.format(
            len(data), data.session_id.nunique(), data.groupby('session_id').size().mean(), data.item_id.nunique(),
            datetime.fromtimestamp(data.timestamp.min()/1000), datetime.fromtimestamp(data.timestamp.max()/1000)
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
        len(data), data.session_id.nunique(), data.item_id.nunique(),
        datetime.fromtimestamp(data.timestamp.min()/1000), datetime.fromtimestamp(data.timestamp.max()/1000)
    ))
    return data


def split_data(data, days_test):
    split_time = data.timestamp.max() - 86400 * 1000 * days_test  # one day = 86400 seconds
    session_max_time = data.groupby('session_id').timestamp.max()

    # split data
    train = data[np.in1d(data.session_id, session_max_time[session_max_time < split_time].index)]
    test = data[np.in1d(data.session_id, session_max_time[session_max_time >= split_time].index)]

    # make sure items in test set have appeared in train set
    test = test[np.in1d(test.item_id, train.item_id)]
    session_length = test.groupby('session_id').size()
    test = test[np.in1d(test.session_id, session_length[session_length >= 2].index)]

    print('split dataset\n\ttrain set events:{}\n\tsessions:{}\n\titems:{}\n\tspan:{} / {}\n\n'.format(
        len(train), train.session_id.nunique(), train.item_id.nunique(),
        datetime.fromtimestamp(train.timestamp.min()/1000), datetime.fromtimestamp(train.timestamp.max()/1000)
    ))
    print('split dataset\n\ttest set events:{}\n\tsessions:{}\n\titems:{}\n\tspan:{} / {}\n\n'.format(
        len(test), test.session_id.nunique(), test.item_id.nunique(),
        datetime.fromtimestamp(test.timestamp.min()/1000), datetime.fromtimestamp(test.timestamp.max()/1000)
    ))
    print('average length:{:.2f}\n\n'.format((len(train)+len(test)) / (train.session_id.nunique()+test.session_id.nunique())))
    return train, test


def trans_data(train, test):
    """renumber items to start from 1 and generate sequences"""
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
