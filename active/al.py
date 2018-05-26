# coding: utf-8
#!/usr/bin/env python3
"""
The script helps guide the users to quickly understand how to use
libact by going through a simple active learning task with clear
descriptions.
"""

import copy
import os

import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

# libact classes
from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.models import *
from libact.query_strategies import *
from libact.labelers import IdealLabeler
# from cp-cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab
from dealword import read_vocab, read_category, batch_iter, process_file, build_vocab


def run(trn_ds, tst_ds, lbr, model, qs, quota):
    E_in, E_out = [], []
    i = 1
    for _ in range(quota):
        # Standard usage of libact objects
        ask_id = qs.make_query()
        print (str(i)+"th times to ask.")
        print (ask_id)
        i = i + 1
        X, _ = zip(*trn_ds.data)
        lb = lbr.label(X[ask_id])
        trn_ds.update(ask_id, lb)

        model.train(trn_ds)
        E_in = np.append(E_in, 1 - model.score(trn_ds))
        E_out = np.append(E_out, 1 - model.score(tst_ds))

    return E_in, E_out


def split_train_test(dataset_filepath, test_size, n_labeled):
    base_dir = 'data/news'
    train_dir = os.path.join(base_dir,'train3_shuf_5000.txt')
    vocab_dir = os.path.join(base_dir,'cnews.vocab_test.txt')
    if not os.path.exists(vocab_dir):
        build_vocab(train_dir,vocab_dir,2000)
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)

    x,y = process_file(train_dir,word_to_id, cat_to_id,600)
    listy = []
    for i in range(np.shape(y)[0]):
        for j in range(np.shape(y)[1]):
            if y[i][j] == 1:
                listy.append(j)

    listy = np.array(listy) 

    # X, y = import_libsvm_sparse(dataset_filepath).format_sklearn()


    X_train, X_test, y_train, y_test = \
        train_test_split(x, listy, test_size=test_size)
    trn_ds = Dataset(X_train, np.concatenate(
        [y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))
    tst_ds = Dataset(X_test, y_test)
    fully_labeled_trn_ds = Dataset(X_train, y_train)
#    print (fully_labeled_trn_ds.get_entries()[0])
    return trn_ds, tst_ds, y_train, fully_labeled_trn_ds


def main():
    # Specifiy the parameters here:
    # path to your binary classification dataset
    base_dir = 'data/cnews'
    train_dir = os.path.join(base_dir,'train3_shuf.txt')
    vocab_dir = os.path.join(base_dir,'cnews.vocab_5000.txt')
    # dataset_filepath = os.path.join(
        # os.path.dirname(os.path.realpath(__file__)), 'diabetes.txt')
    test_size = 0.33    # the percentage of samples in the dataset that will be
    # randomly selected and assigned to the test set
    n_labeled = 1000      # number of samples that are initially labeled

    # Load dataset
    trn_ds, tst_ds, y_train, fully_labeled_trn_ds = \
        split_train_test(train_dir, test_size, n_labeled)
    trn_ds2 = copy.deepcopy(trn_ds)
    lbr = IdealLabeler(fully_labeled_trn_ds)

    quota = len(y_train) - n_labeled    # number of samples to query

    # Comparing UncertaintySampling strategy with RandomSampling.
    # model is the base learner, e.g. LogisticRegression, SVM ... etc.
    qs = UncertaintySampling(trn_ds, method='lc', model=LogisticRegression())
    model = LogisticRegression()
    E_in_1, E_out_1 = run(trn_ds, tst_ds, lbr, model, qs, quota)

    qs2 = RandomSampling(trn_ds2)
    model = LogisticRegression()
    E_in_2, E_out_2 = run(trn_ds2, tst_ds, lbr, model, qs2, quota)

    # Plot the learning curve of UncertaintySampling to RandomSampling
    # The x-axis is the number of queries, and the y-axis is the corresponding
    # error rate.
    query_num = np.arange(1, quota + 1)
    plt.plot(query_num, E_in_1, 'b', label='qs Ein')
    plt.plot(query_num, E_in_2, 'r', label='random Ein')
    plt.plot(query_num, E_out_1, 'g', label='qs Eout')
    plt.plot(query_num, E_out_2, 'k', label='random Eout')
    plt.xlabel('Number of Queries')
    plt.ylabel('Error')
    plt.title('Experiment Result')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=5)
    plt.savefig('result.png')
    plt.show()


if __name__ == '__main__':
    main()
