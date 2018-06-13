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
try:
    from data.dealwordindict import read_vocab, read_category, batch_iter, process_file, build_vocab
except Exception: #ImportError
    from dealwordindict import read_vocab, read_category, batch_iter, process_file, build_vocab
# from dealwordindict import read_vocab, read_category, batch_iter, process_file, build_vocab
import time
import heapq
from data.rnnmodel import RNN_Probability_Model
import random
# import zip

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
        
        start_time = time.time()
        ask_id = qs.make_query()
        print (str(i)+"th times to ask.======================")
        time_dif = get_time_dif(start_time)

        i = i + 1
        X, _ = zip(*trn_ds.data)
        lb = lbr.label(X[ask_id])
        trn_ds.update(ask_id,lb)
        model.train(trn_ds)
        
        E_in = np.append(E_in, 1 - model.score(trn_ds))
        E_out = np.append(E_out, 1 - model.score(tst_ds))
    return E_in, E_out


def realrun_random(trn_ds, tst_ds, lbr, model, qs, quota):
    E_in, E_out = [], []
    intern = 0
    finalnum = 0
    if quota % 8 == 0:
        intern = int(quota / 8)
    else:
        intern = int(quota / 8) + 1
        finalnum = int(quota % 8)

    for t in range(intern):
        print ("[Random]this is the "+str(t)+" times to ask")

        # scores = model.predict_pro(trn_ds)
        # unlabeled_entry_ids, X_pool = zip(*trn_ds.get_unlabeled_entries())
        # first, scores = qs.make_query(return_score=True)

        # python 2&&python3
        # number, num_score = zip(*scores)[0], zip(*scores)[1]
        # num_score = next(zip*scores)
        # itscore = zip(*scores)
        # number = next(itscore)
        # num_score = next(itscore)

        # num_score_array = np.array(num_score)
        # max_n = heapq.nlargest(quota, range(len(num_score_array)), num_score_array.take)
        unlabeled_entry_ids, X_pool = zip(*trn_ds.get_unlabeled_entries())
        if t == intern - 1 and finalnum != 0:
            max_n = random.sample(unlabeled_entry_ids,finalnum)
        else:
            max_n = random.sample(unlabeled_entry_ids, 8)

        X, _ = zip(*trn_ds.data)
        # print (max_n)
        for ask_id in max_n:
            lb = lbr.label(X[ask_id])
            trn_ds.update(ask_id, lb)

        model.train(trn_ds)

        # E_in = np.append(E_in, 1 - model.score(trn_ds))
        E_out = np.append(E_out, 1 - model.score(tst_ds))
        # print (E_in)
        # print (E_out)

    return E_out
def realrun_qs(trn_ds, tst_ds, lbr, model,qs, quota):
    E_in, E_out = [], []
    intern = 0
    finalnum = 0
    if quota % 8 == 0:
        intern = int( quota / 8)
    else:
        intern = int(quota / 8) + 1
        finalnum = int(quota % 8)

    for t in range(intern):
        print ("[QS]this is the "+str(t)+" times to ask")

        # scores = model.predict_pro(trn_ds)
        # unlabeled_entry_ids, X_pool = zip(*trn_ds.get_unlabeled_entries())
        first, scores = qs.make_query(return_score=True)

        #python 2&&python3
        #number, num_score = zip(*scores)[0], zip(*scores)[1]
        # num_score = next(zip*scores)
        itscore = zip(*scores)
        number = next(itscore)
        num_score = next(itscore)

        # print (number)
        # print (num_score)
        num_score_array = np.array(num_score)
        # max_n = heapq.nlargest(quota, range(len(num_score_array)), num_score_array.take)

        if t == intern - 1 and finalnum != 0:
            max_n = heapq.nlargest(finalnum, range(len(num_score_array)), num_score_array.take)
        else:
            max_n = heapq.nlargest(8, range(len(num_score_array)), num_score_array.take)

        unlabeled_entry_ids, X_pool = zip(*trn_ds.get_unlabeled_entries())
        print (unlabeled_entry_ids)

        X, _ = zip(*trn_ds.data)
        # print (max_n)
        for ask_id in max_n:
            real_id = unlabeled_entry_ids[ask_id]
            lb = lbr.label(X[real_id])
            trn_ds.update(real_id, lb)

        model.train(trn_ds)

        # E_in = np.append(E_in, 1 - model.score(trn_ds))
        E_out = np.append(E_out, 1 - model.score(tst_ds))
        # print (E_in)
        # print (E_out)

    return E_out

def runrnn(trn_ds, tst_ds, val_ds, lbr, model, quota, best_val):
    E_in, E_out = [], []
    intern = 0
    finalnum = 0
    if quota % 8 == 0:
        intern = int( quota / 8)
    else:
        intern = int(quota / 8) + 1
        finalnum = int(quota % 8)

    for t in range(intern):
        print ("[RNN]this is the "+str(t)+" times to ask")

        scores = model.predict_pro(trn_ds)
        # first, scores = qs.make_query(return_score=True)
        unlabeled_entry_ids, X_pool = zip(*trn_ds.get_unlabeled_entries())
        # number, num_score = zip(*scores)[0], zip(*scores)[1]
        # num_score_array = np.array(num_score)
        # max_n = headq.nlargest(8,num_score_array)
        if t == intern - 1 and finalnum != 0:
            max_n = heapq.nsmallest(finalnum, range(len(scores)), scores.take)
        else:
            max_n = heapq.nsmallest(8, range(len(scores)), scores.take)


        # print (max_n)
        X, _ = zip(*trn_ds.data)

        print (max_n)
        for ask_id in max_n:
            real_id = unlabeled_entry_ids[ask_id]
            lb = lbr.label(X[real_id])
            trn_ds.update(real_id, lb)


        best_val = model.retrain(trn_ds, val_ds, best_val)

        # E_in = np.append(E_in, 1 - model.score(trn_ds))
        E_out = np.append(E_out, 1 - model.score(tst_ds))
        # print (E_in)
        print (E_out)

    return E_out


def split_train_test(dataset_filepath, test_size, n_labeled):
    #base_dir = './data/yinan'
    #train_dir = os.path.join(base_dir,'labeled.txt')
    #vocab_dir = os.path.join(base_dir,'vocab_yinan_1.txt')
    train_dir = './data/labeled1.txt'
    vocab_dir = './data/vocab_yinan_test_rnn4.txt'
    if not os.path.exists(vocab_dir):
        build_vocab(train_dir,vocab_dir,1000)
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)

    x,y = process_file(train_dir,word_to_id, cat_to_id,600)
    listy = []
    for i in range(np.shape(y)[0]):
        for j in range(np.shape(y)[1]):
            if y[i][j] == 1:
                listy.append(j)
    listy = np.array(listy) 


    X_train, X_test, y_train, y_test = \
        train_test_split(x, listy, test_size=test_size)

    # X_train = X_train[:(n_labeled + 24)]
    trn_ds = Dataset(X_train, np.concatenate(
        [y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))
    # trn_ds = Dataset(X_train, np.concatenate(
    #     [y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))
    fully_tst_ds = Dataset(X_test, y_test)

    X_val, X_real_test, y_val, y_real_test = \
            train_test_split(X_test, y_test, test_size=0.5)

    tst_ds = Dataset(X_real_test, y_real_test)
    val_ds = Dataset(X_val, y_val)

    # fully_tst_ds = Dataset(X_test, y_test)


    fully_labeled_trn_ds = Dataset(X_train, y_train)
#    print (fully_labeled_trn_ds.get_entries()[0])
    return trn_ds, tst_ds, y_train, fully_labeled_trn_ds, fully_tst_ds, val_ds



def main():
    # Specifiy the parameters here:
    # path to your binary classification dataset
    base_dir = 'data/yinan'
    train_dir = os.path.join(base_dir,'labeled1.txt')
    vocab_dir = os.path.join(base_dir,'vocab_yinan_4.txt')
    test_size = 0.3    # the percentage of samples in the dataset that will be
    n_labeled = 300     # number of samples that are initially labeled

    result = {'E1':[],'E2':[],'E3':[]}
    for i in range(1):
        trn_ds, tst_ds, y_train, fully_labeled_trn_ds,fully_tst_ds,val_ds = \
         split_train_test(train_dir, test_size, n_labeled)
        trn_ds2 = copy.deepcopy(trn_ds)
        trn_ds3 = copy.deepcopy(trn_ds)
        lbr = IdealLabeler(fully_labeled_trn_ds)


        quota = len(y_train) - n_labeled

        # quota = 24
        print (len(trn_ds3.get_labeled_entries()))
        print(len(fully_tst_ds.get_labeled_entries()))
        print(len(tst_ds.get_labeled_entries()))
        print(len(val_ds.get_labeled_entries()))

        # result['E1'].append(E_out_1)
        model = SVM(kernel='rbf', decision_function_shape='ovr')
        qs2 = RandomSampling(trn_ds2)
        E_out_2 = realrun_random(trn_ds2, fully_tst_ds, lbr, model, qs2, quota)



        qs = UncertaintySampling(trn_ds, method='sm',model=SVM(decision_function_shape='ovr'))
        model = SVM(kernel='rbf',decision_function_shape='ovr')
        E_out_1 = realrun_qs(trn_ds, fully_tst_ds, lbr, model, qs, quota)

        modelrnn = RNN_Probability_Model()
        modelrnn.train(trn_ds3, val_ds)
        test_acc = modelrnn.test(tst_ds)
        E_out_3 = runrnn(trn_ds3, tst_ds, val_ds, lbr, modelrnn, quota, test_acc)
        # test_acc = modelrnn.test(tst_ds)

        result['E1'].append(E_out_1)
        result['E2'].append(E_out_2)

        result['E3'].append(E_out_3)



    # E_out_1 = np.mean(result['E1'],axis=0)
    E_out_2 = np.mean(result['E2'],axis=0)
    E_out_3 = np.mean(result['E3'],axis=0)
    # Plot the learning curve of UncertaintySampling to RandomSampling
    # The x-axis is the number of queries, and the y-axis is the corresponding
    # error rate.
    print (E_out_1)
    print (E_out_2)
    print (E_out_3)
    if quota % 8 == 0:
        intern = int( quota / 8)
    else:
        intern = int(quota / 8) + 1
    query_num = np.arange(1, intern + 1)
    plt.figure(figsize=(10,8))
    #plt.plot(query_num, E_in_1, 'b', label='qs Ein')
    #plt.plot(query_num, E_in_2, 'r', label='random Ein')
    plt.plot(query_num, E_out_1, 'g', label='qs Eout')
    plt.plot(query_num, E_out_2, 'k', label='random Eout')
    plt.plot(query_num, E_out_3, 'r', label='rnn Eout')
    plt.xlabel('Number of Queries')
    plt.ylabel('Error')
    plt.title('Experiment Result')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=5)
    plt.savefig('testmerge_rnn.png')
    plt.show()


if __name__ == '__main__':
    main()
