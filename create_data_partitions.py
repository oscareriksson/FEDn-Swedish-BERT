import os
import sys
from datasets import load_dataset
from math import floor


def import_data(test_ratio):
    """ Download data. """
    data = load_dataset('swedish_reviews', split='train')
    data = data.to_pandas()
    data = data.sample(frac=1).reset_index(drop=True)
    data.to_csv("data.csv", index=False)
    num_test = int(test_ratio*data.shape[0])
    test_set = data[:num_test]
    train_set = data[num_test:]
    return train_set, test_set


def split_set(dataset, parts):
    n = dataset.shape[0]
    local_n = floor(n/parts)
    result = [dataset[i*local_n: (i+1)*local_n] for i in range(parts)]
    return result


if __name__ == '__main__':

    if len(sys.argv) < 2:
        nr_of_datasets = 2
    else:
        nr_of_datasets = int(sys.argv[1])

    train_set, test_set = import_data(test_ratio=0.1)
    print('train_set', len(train_set))
    print('test_set', len(test_set))

    train_sets = split_set(train_set, nr_of_datasets)
    test_sets = split_set(test_set, nr_of_datasets)

    if not os.path.exists('data'):
        os.mkdir('data')
    if not os.path.exists('data/clients'):
        os.mkdir('data/clients')

    for i in range(nr_of_datasets):
        if not os.path.exists('data/clients/{}'.format(str(i))):
            os.mkdir('data/clients/{}'.format(str(i)))
        train_sets[i].to_csv('data/clients/{}'.format(str(i)) + '/train.csv', index=False)
        test_sets[i].to_csv('data/clients/{}'.format(str(i)) + '/test.csv', index=False)