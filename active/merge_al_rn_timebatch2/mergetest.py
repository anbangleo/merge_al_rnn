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
import tensorflow.contrib.keras as kr
# from cp-cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab
try:
    from data.dealwordindict import read_vocab, read_category, batch_iter, process_file, process_file_rnn, build_vocab
except Exception: #ImportError
    from dealwordindict import read_vocab, read_category, batch_iter, process_file, process_file_rnn, build_vocab
# from dealwordindict import read_vocab, read_category, batch_iter, process_file, build_vocab
import time
from datetime import timedelta
import heapq
from data.rnnmodel import RNN_Probability_Model,TRNNConfig
import random

# import zip

from datetime import timedelta
import gc
np.set_printoptions(threshold=1e6)
def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(microseconds=int(round(time_dif*1000)))


def realrun_random(trn_ds, tst_ds, lbr, model, qs, quota, batchsize):
    E_in, E_out = [], []
    intern = 0
    finalnum = 0
    print ("[Important] Start the Random Train:")
    start_time = time.time()
    if quota % batchsize == 0:
        intern = int(quota / batchsize)
    else:
        intern = int(quota / batchsize ) + 1
        finalnum = int(quota % batchsize)

    for t in range(intern):
        print ("[Random]this is the "+str(t)+" times to ask")
        unlabeled_entry_ids, X_pool = zip(*trn_ds.get_unlabeled_entries())
        if t == intern - 1 and finalnum != 0:
            max_n = random.sample(unlabeled_entry_ids,finalnum)
        else:
            max_n = random.sample(unlabeled_entry_ids, batchsize)

        X, _ = zip(*trn_ds.data)
        for ask_id in max_n:
            lb = lbr.label(X[ask_id])
            trn_ds.update(ask_id, lb)

        model.train(trn_ds)
        E_out = np.append(E_out, model.score(tst_ds))
        print (E_out)

    E_time = get_time_dif(start_time)

    return E_out, E_time
def realrun_qs(trn_ds, tst_ds, lbr, model,qs, quota, batchsize):
    E_in, E_out = [], []
    E_time = []
    intern = 0
    finalnum = 0
    print ("[Important] Start the UncertaintySampling Train:")

    if quota % batchsize == 0:
        intern = int( quota / batchsize)
    else:
        intern = int(quota / batchsize) + 1
        finalnum = int(quota % batchsize)

    for t in range(intern):
        print ("[QS]this is the "+str(t)+" times to ask")
        start_time = time.time()

        # scores = model.predict_pro(trn_ds)
        # unlabeled_entry_ids, X_pool = zip(*trn_ds.get_unlabeled_entries())
        first, scores = qs.make_query(return_score=True)
        itscore = zip(*scores)
        number = next(itscore)
        num_score = next(itscore)

        num_score_array = np.array(num_score)

        if t == intern - 1 and finalnum != 0:
            max_n = heapq.nlargest(finalnum, range(len(num_score_array)), num_score_array.take)
        else:
            max_n = heapq.nlargest(batchsize, range(len(num_score_array)), num_score_array.take)

        unlabeled_entry_ids, X_pool = zip(*trn_ds.get_unlabeled_entries())

        X, _ = zip(*trn_ds.data)
        # print (max_n)
        for ask_id in max_n:
            real_id = unlabeled_entry_ids[ask_id]
            lb = lbr.label(X[real_id])
            trn_ds.update(real_id, lb)

        model.train(trn_ds)

        # E_in = np.append(E_in, 1 - model.score(trn_ds))
        E_out = np.append(E_out, model.score(tst_ds))
        # print (E_in)
        Etime = get_time_dif(start_time)
        E_time.append(str(Etime))
        # print (Etime)
        # print (E_time)

        print (E_out)



    return E_out, E_time

def runrnn(trn_ds, tst_ds, val_ds, lbr, model, quota, best_val, batchsize):
    E_in, E_out = [], []
    intern = 0
    finalnum = 0
    print ("[Important] Start the RNN Train:")
    start_time = time.time()
    if quota % batchsize == 0:
        intern = int( quota / batchsize)
    else:
        intern = int(quota / batchsize) + 1
        finalnum = int(quota % batchsize)

    for t in range(intern):
        print ("[RNN]this is the "+str(t)+" times to ask")
        x_first_train = []
        y_first_train = []

        scores = model.predict_pro(trn_ds)

        unlabeled_entry_ids, X_pool = zip(*trn_ds.get_unlabeled_entries())

        if t == intern - 1 and finalnum != 0:
            max_n = heapq.nsmallest(finalnum, range(len(scores)), scores.take)
        else:
            max_n = heapq.nsmallest(batchsize, range(len(scores)), scores.take)


        X, _ = zip(*trn_ds.data)

        print (max_n)
        for ask_id in max_n:
            real_id = unlabeled_entry_ids[ask_id]
            lb = lbr.label(X[real_id])
            trn_ds.update(real_id, lb)
            x_first_train.append(X[real_id])
            y_first_train.append(lb)

        x_first_train = np.array(x_first_train)
        y_first_train = np.array(y_first_train)

        first_train = Dataset(x_first_train,y_first_train)

        best_val = model.retrain(trn_ds, val_ds, best_val, first_train)

        # E_in = np.append(E_in, 1 - model.score(trn_ds))
        E_out = np.append(E_out, model.score(tst_ds))
        # print (E_in)
        print (E_out)

    E_time = get_time_dif(start_time)

    return E_out, E_time


def split_train_test(train_dir, vocab_dir, test_size, n_labeled, wordslength):
    #train_dir = './data/labeled1.txt'
    #vocab_dir = './data/vocab_yinan_test_rnn4.txt'
    if not os.path.exists(vocab_dir):
        build_vocab(train_dir,vocab_dir,1000)
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)

    x,y = process_file(train_dir, word_to_id, cat_to_id, wordslength)
    # x_rnn, y_rnn = process_file_rnn(train_dir, word_to_id, cat_to_id, 600)

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

    fully_labeled_trn_ds = Dataset(X_train, y_train)
#    print (fully_labeled_trn_ds.get_entries()[0])
    return trn_ds, tst_ds, y_train, fully_labeled_trn_ds, fully_tst_ds, val_ds

def split_train_test_rnn(train_dir, vocab_dir, vocab_size, test_size, val_size, n_labeled, wordslength, categories_class):
    if not os.path.exists(vocab_dir):
        build_vocab(train_dir,vocab_dir, vocab_size)
    categories, cat_to_id = read_category(categories_class)
    words, word_to_id = read_vocab(vocab_dir)

    data_id, label_id = process_file_rnn(train_dir, word_to_id, cat_to_id,wordslength)
    # x_rnn, y_rnn = process_file_rnn(train_dir, word_to_id, cat_to_id, 600)

    y = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))
    listy = []
    for i in range(np.shape(y)[0]):
        for j in range(np.shape(y)[1]):
            if y[i][j] == 1:
                listy.append(j)
    listy = np.array(listy)


    X_train, X_test, y_train, y_test = \
        train_test_split(data_id, listy, test_size=test_size)


    X_train_al = []
    X_test_al = []
    res = []
    for i in X_train:
        for j in range(wordslength):
            a = i.count(j)
            if a > 0:
                res.append(a)
            else:
                res.append(0)
        X_train_al.append(res)
        res = []

    for i in X_test:
        for j in range(wordslength):
            a = i.count(j)
            if a > 0:
                res.append(a)
            else:
                res.append(0)
        X_test_al.append(res)
        res = []


    X_train_al = np.array(X_train_al)
    X_test_al = np.array(X_test_al)


    trn_ds_al = Dataset(X_train_al, np.concatenate(
        [y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))

    tst_ds_al = Dataset(X_test_al, y_test)


    X_train_rnn = kr.preprocessing.sequence.pad_sequences(X_train, wordslength)
    X_test_rnn = kr.preprocessing.sequence.pad_sequences(X_test, wordslength)

    X_train_rnn, X_val_rnn, y_train_rnn, y_val_rnn = \
            train_test_split(X_train_rnn, y_train, test_size=val_size)

    trn_ds_rnn = Dataset(X_train_rnn, np.concatenate(
        [y_train_rnn[:n_labeled], [None] * (len(y_train_rnn) - n_labeled)]))

    val_ds_rnn = Dataset(X_val_rnn, y_val_rnn)

    tst_ds_rnn = Dataset(X_test_rnn, y_test)



    fully_labeled_trn_ds_al = Dataset(X_train_al, y_train)
    fully_labeled_trn_ds_rnn = Dataset(X_train_rnn, y_train_rnn)

    return trn_ds_al, tst_ds_al, y_train_rnn, fully_labeled_trn_ds_al, \
        trn_ds_rnn, tst_ds_rnn, fully_labeled_trn_ds_rnn, val_ds_rnn


def main():
    config = TRNNConfig()

    train_dir = './data/train10_shuf_3000.txt'
    vocab_dir = './data/vocab_train10_shuf_3000.txt'
    batchsize = 64
    wordslength = config.seq_length
    vocab_size = config.vocab_size
    numclass = config.num_classes
    val_size = 0.15
    test_size = 0.2    # the percentage of samples in the dataset that will be
    n_labeled = 300     # number of samples that are initially labeled
    categories_class = ['体育', '家居', '娱乐','游戏','财经','房产','教育','时尚','时政','科技']
    batch_one = 1
    batch_sixteen = 16
    batch_128 = 128
    batch_256 = 256
    resultfile = open('queryresult4.txt','w')
    result = {'E1':[],'E2':[],'E3':[]}
    for i in range(1):
        trn_ds_al, tst_ds_al, y_train_rnn, fully_labeled_trn_ds_al, trn_ds_rnn, tst_ds_rnn, fully_labeled_trn_ds_rnn, val_ds_rnn = \
         split_train_test_rnn(train_dir, vocab_dir, vocab_size, test_size, val_size, n_labeled, wordslength, categories_class)
        trn_ds2 = copy.deepcopy(trn_ds_al)
        trn_ds3 = copy.deepcopy(trn_ds_al)
        trn_ds4 = copy.deepcopy(trn_ds_al)

        trn_ds5 = copy.deepcopy(trn_ds_al)
        trn_ds6 = copy.deepcopy(trn_ds_al)

        lbr_al = IdealLabeler(fully_labeled_trn_ds_al)
        lbr_rnn = IdealLabeler(fully_labeled_trn_ds_rnn)


        quota = len(y_train_rnn) - n_labeled

        model = SVM(kernel='rbf', decision_function_shape='ovr')
        qs2 = UncertaintySampling(trn_ds_al, method='sm', model=SVM(decision_function_shape='ovr'))
        E_out_us16, E_time_us16 = realrun_qs(trn_ds_al, tst_ds_al, lbr_al, model, qs2, quota, batch_sixteen)


        qs = UncertaintySampling(trn_ds3, method='sm',model=SVM(decision_function_shape='ovr'))
        model = SVM(kernel='rbf',decision_function_shape='ovr')
        E_out_us64, E_time_us64 = realrun_qs(trn_ds3, tst_ds_al, lbr_al, model, qs, quota, batchsize)

        qs4 = UncertaintySampling(trn_ds4, method='sm',model=SVM(decision_function_shape='ovr'))
        model = SVM(kernel='rbf',decision_function_shape='ovr')
        E_out_us1, E_time_us1 = realrun_qs(trn_ds4, tst_ds_al, lbr_al, model, qs4, quota, batch_one)

        qs5 = UncertaintySampling(trn_ds5, method='sm',model=SVM(decision_function_shape='ovr'))
        model = SVM(kernel='rbf',decision_function_shape='ovr')
        E_out_us128, E_time_us128 = realrun_qs(trn_ds5, tst_ds_al, lbr_al, model, qs5, quota, batch_128)

        qs6 = UncertaintySampling(trn_ds6, method='sm',model=SVM(decision_function_shape='ovr'))
        model = SVM(kernel='rbf',decision_function_shape='ovr')
        E_out_us256, E_time_us256 = realrun_qs(trn_ds6, tst_ds_al, lbr_al, model, qs5, quota, batch_256)




        resultfile.writelines(str(E_out_us1)+'\n')
        resultfile.writelines(str(E_time_us1)+'\n')
        resultfile.writelines(str(E_out_us16)+'\n')
        resultfile.writelines(str(E_time_us16)+'\n')
        resultfile.writelines(str(E_out_us64)+'\n')
        resultfile.writelines(str(E_time_us64)+'\n')
        resultfile.writelines(str(E_out_us128)+'\n')
        resultfile.writelines(str(E_time_us128)+'\n')
        resultfile.writelines(str(E_out_us256)+'\n')
        resultfile.writelines(str(E_time_us256)+'\n')
      #  if len(E_out_us1) > len(E_out_us16):
      #      E_out_us1.pop()
      #  if len(E_out_us1) > len(E_out_us64):
      #      E_out_us1.pop()
        # test_acc = modelrnn.test(tst_ds)
        for t in range(len(E_out_us1)):
            if t % batchsize == 0:
                result['E1'].append(E_out_us1[t])

        for m in range(len(E_out_us16)):
            if m % 4 == 0:
                result['E3'].append(E_out_us16[m])
        # result['E3'].append(E_out_rnn1)
        result['E2'].append(E_out_us64)


    E_out_us1 = np.mean(result['E1'],axis=0)
    E_out_us64 = np.mean(result['E2'],axis=0)
    E_out_us16 = np.mean(result['E3'],axis=0)
    # Plot the learning curve of UncertaintySampling to RandomSampling
    # The x-axis is the number of queries, and the y-axis is the corresponding
    # error rate.
    print (np.shape(E_out_us1))
    print (np.shape(E_out_us16))
    print (np.shape(E_out_us64))
    
    print ("[Result] for Uncertainty Sampling")
    print (E_out_us1)
    print (E_time_us1)
    print (E_out_us16)
    print (E_time_us16)
    print (E_out_us64)
    print (E_time_us64)
    if quota % batchsize == 0:
        intern = int( quota / batchsize)
    else:
        intern = int(quota / batchsize) + 1
    query_num = np.arange(1, intern + 1)
    plt.figure(figsize=(10,8))
    plt.plot(query_num, E_out_us1, 'b', label='Single1')
    plt.plot(query_num, E_out_us16, 'r', label='Batch16')
    plt.plot(query_num, E_out_us64, 'g', label='Batch64')
    plt.xlabel('Number of Batches')
    plt.ylabel('Accuracy')
    plt.title('Result')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=5)
    plt.savefig('testmerge_rnn_10_3000_0705_time.png')
    plt.show()


if __name__ == '__main__':
    main()