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
import matplotlib
matplotlib.use('Agg')
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
from dealwordindict import read_vocab, read_category, batch_iter, process_file, build_vocab
import time
from datetime import timedelta
import gc

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return time_dif
    # return timedelta(seconds=int(round(time_dif)))

def run(trn_ds, tst_ds, lbr, model, qs, quota):
    E_in, E_out = [], []
    i = 1
    for _ in range(quota):
       # gc.disable()
       # Standard usage of libact objects
        start_time = time.time()
        ask_id = qs.make_query()
        print (str(i)+"th times to ask.======================")
        print (ask_id)
        time_dif = get_time_dif(start_time)
        print ("time to ask"+str(time_dif))        

        i = i + 1
        X, _ = zip(*trn_ds.data)

        lb = lbr.label(X[ask_id])
        trn_ds.update(ask_id, lb)
        start_time = time.time()
        model.train(trn_ds)
        time_dif = get_time_dif(start_time)
        print ("time to train"+str(time_dif))
       # gc.enable()
        E_in = np.append(E_in, 1 - model.score(trn_ds))
        E_out = np.append(E_out, 1 - model.score(tst_ds))
    return E_in, E_out


def split_train_test(dataset_filepath, test_size, n_labeled):
    #base_dir = './data/yinan'
    #train_dir = os.path.join(base_dir,'labeled.txt')
    #vocab_dir = os.path.join(base_dir,'vocab_yinan_1.txt')
    train_dir = '/home/ab/test/al/active/data/yinan/labeled1.txt'
    categories, cat_to_id = read_category()

    x,y = process_file(train_dir, cat_to_id,200)
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
    base_dir = 'data/yinan'
    train_dir = os.path.join(base_dir,'labeled1.txt')
    vocab_dir = os.path.join(base_dir,'vocab_yinan_3.txt')
    # dataset_filepath = os.path.join(
        # os.path.dirname(os.path.realpath(__file__)), 'diabetes.txt')
    test_size = 0.3    # the percentage of samples in the dataset that will be
    # randomly selected and assigned to the test set
    n_labeled = 20      # number of samples that are initially labeled

    result = {'E1':[],'E2':[]}
    for i in range(2):
    # Load datas
        trn_ds, tst_ds, y_train, fully_labeled_trn_ds = \
         split_train_test(train_dir, test_size, n_labeled)
        trn_ds2 = copy.deepcopy(trn_ds)
        lbr = IdealLabeler(fully_labeled_trn_ds)

        #quota = len(y_train) - n_labeled    # number of samples to query
        quota = 680
    # Comparing UncertaintySampling strategy with RandomSampling.
    # model is the base learner, e.g. LogisticRegression, SVM ... etc.
        model = SVM(kernel = 'rbf',decision_function_shape='ovr')
        qs = UncertaintySampling(trn_ds, method='sm',model=SVM(decision_function_shape='ovr'))
        E_in_1, E_out_1 = run(trn_ds, tst_ds, lbr, model, qs, quota)
        result['E1'].append(E_out_1)
        qs2 = RandomSampling(trn_ds2)
        E_in_2, E_out_2 = run(trn_ds2, tst_ds, lbr, model, qs2, quota)
        result['E2'].append(E_out_2)
    E_out_1 = np.mean(result['E1'],axis=0)
    E_out_2 = np.mean(result['E2'],axis=0)
    # Plot the learning curve of UncertaintySampling to RandomSampling
    # The x-axis is the number of queries, and the y-axis is the corresponding
    # error rate.
    query_num = np.arange(1, quota + 1)
    plt.figure(figsize=(10,8))
    #plt.plot(query_num, E_in_1, 'b', label='qs Ein')
    #plt.plot(query_num, E_in_2, 'r', label='random Ein')
    plt.plot(query_num, E_out_1, 'g', label='qs Eout')
    plt.plot(query_num, E_out_2, 'k', label='random Eout')
    plt.xlabel('Number of Queries')
    plt.ylabel('Error')
    plt.title('Experiment Result')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=5)
    plt.savefig('resultlg_features.png')
    #plt.show()


if __name__ == '__main__':
    main()
