# coding: utf-8
#!/usr/bin/env python3
"""
The script helps guide the users to quickly understand how to use
libact by going through a simple active learning task with clear
descriptions.
"""

import matplotlib
import numpy as np

matplotlib.use('Agg')

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

# libact classes
# from cp-cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab


#from active.merge_al_rnn.data.dealwordindict import read_vocab, read_category, batch_iter, build_vocab
from .dealwordindict import read_vocab, read_category, batch_iter, build_vocab
import time
from datetime import timedelta

import tensorflow as tf
import tensorflow.contrib.keras as kr

from .rnn_model import TRNNConfig, TextRNN
#from active.merge_al_rnn.rnn_model import TRNNConfig, TextRNN
# from .cnews_loader import read_category, read_vocab
from sklearn.metrics import accuracy_score

from libact.query_strategies import *
from sklearn import metrics
try:
    bool(type(unicode))
except NameError:
    unicode = str




class RNN_Probability_Model:
    """docstring for RNNmodel"""
    def __init__(self):
        # self.train_dir = '/home/ab/test/al/active/data/yinan/labeled1.txt'
        self.vocab_dir = '/home/ab/test/al/active/data/yinan/vocab_yinan_test_rnn3.txt'
        # self.base_dir = '/home/ab/test/al/active/data/yinan/rnn/testrnn/'
        # self.train_dir = os.path.join(self.base_dir, 'train.txt')
        # # self.test_dir = os.path.join(self.base_dir, 'test.txt')
        # # self.val_dir = os.path.join(self.base_dir, 'val.txt')
        # self.vocab_dir = os.path.join(self.base_dir, 'vocab1_jieba.txt')

        self.save_dir = 'checkpoints/textmergernn'
        self.save_path = os.path.join(self.save_dir, 'best_validation')  # 最佳验证结果保存路径
        self.config = TRNNConfig()
        # if not os.path.exists(self.vocab_dir):  # 如果不存在词汇表，重建
        #     build_vocab(self.train_dir, self.vocab_dir, self.config.vocab_size)
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(self.vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextRNN(self.config)


        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())


    def get_time_dif(self,start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

    def feed_data(self, x_batch, y_batch, keep_prob):
        feed_dict = {
            self.model.input_x: x_batch,
            self.model.input_y: y_batch,
            self.model.keep_prob: keep_prob
        }
        return feed_dict

    def evaluate(self, sess, x_, y_):
        """评估在某一数据上的准确率和损失"""
        data_len = len(x_)
        batch_eval = batch_iter(x_, y_, 8)
        total_loss = 0.0
        total_acc = 0.0
        for x_batch, y_batch in batch_eval:
            batch_len = len(x_batch)
            feed_dict = self.feed_data(x_batch, y_batch, 1.0)
            loss, acc = sess.run([self.model.loss, self.model.acc], feed_dict=feed_dict)
            total_loss += loss * batch_len
            total_acc += acc * batch_len

        return total_loss / data_len, total_acc / data_len

    def train(self,trn_dataset,val_dataset):
        # newdataset = dataset.format_sklearn()
        print("Configuring TensorBoard and Saver...")
        # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
        tensorboard_dir = 'tensorboard/textmergernn'
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)

        tf.summary.scalar("loss", self.model.loss)
        tf.summary.scalar("accuracy", self.model.acc)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(tensorboard_dir)

        # 配置 Saver
        saver = tf.train.Saver()
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        print("Loading training and validation data...")
        # 载入训练集与验证集
        start_time = time.time()

        # x_train, y_train = process_al_file(trn_dataset, self.word_to_id, self.cat_to_id, self.config.seq_length)
        # x_val, y_val = process_al_file(val_dataset, self.word_to_id, self.cat_to_id, self.config.seq_length)
        # print (trn_dataset)
        # print (len(trn_dataset.format_sklearn()))
        x_train, y_train = trn_dataset.format_sklearn()
        y_train = kr.utils.to_categorical(y_train, num_classes=3)
        # print (np.shape(x_train))
        # print (np.shape(y_train))
        print (y_train)

        x_val, y_val = val_dataset.format_sklearn()
        y_val = kr.utils.to_categorical(y_val, num_classes=3)

        time_dif = self.get_time_dif(start_time)
        print("Time usage:", time_dif)

        # 创建session
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        writer.add_graph(session.graph)

        print('Training and evaluating...')
        start_time = time.time()
        total_batch = 0  # 总批次
        best_acc_val = 0.0  # 最佳验证集准确率
        last_improved = 0  # 记录上一次提升批次
        require_improvement = 50  # 如果超过1000轮未提升，提前结束训练

        flag = False
        for epoch in range(self.config.num_epochs):
            print('Epoch:', epoch + 1)
            batch_train = batch_iter(x_train, y_train, self.config.batch_size)
            for x_batch, y_batch in batch_train:
                feed_dict = self.feed_data(x_batch, y_batch, self.config.dropout_keep_prob)
                
                if total_batch % self.config.save_per_batch == 0:
                    # 每多少轮次将训练结果写入tensorboard scalar
                    s = session.run(merged_summary, feed_dict=feed_dict)
                    writer.add_summary(s, total_batch)

                if total_batch % self.config.print_per_batch == 0:
                    # 每多少轮次输出在训练集和验证集上的性能
                    feed_dict[self.model.keep_prob] = 1.0
                    loss_train, acc_train = session.run([self.model.loss, self.model.acc], feed_dict=feed_dict)
                    loss_val, acc_val = self.evaluate(session, x_val, y_val)  # todo

                    if acc_val > best_acc_val:
                        # 保存最好结果
                        best_acc_val = acc_val
                        last_improved = total_batch
                        saver.save(sess=session, save_path=self.save_path)
                        improved_str = '*'
                    else:
                        improved_str = ''

                    time_dif = self.get_time_dif(start_time)
                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                          + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                    print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

                session.run(self.model.optim, feed_dict=feed_dict)  # 运行优化
                total_batch += 1

                if total_batch - last_improved > require_improvement:
                    # 验证集正确率长期不提升，提前结束训练
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break  # 跳出循环
            if flag:  # 同上
                break
        return best_acc_val


    def retrain(self, trn_dataset, val_dataset, best_acc_val):
        tensorboard_dir = 'tensorboard/textmergernn'

        x_train, y_train = trn_dataset.format_sklearn()
        y_train = kr.utils.to_categorical(y_train,num_classes=3)
        x_val, y_val = val_dataset.format_sklearn()
        y_val = kr.utils.to_categorical(y_val,num_classes=3)
        # newdataset = newdataset.format_sklearn()
        # x_train, y_train = process_file(newdataset, self.word_to_id, self.cat_to_id, self.config.seq_length)
        tf.summary.scalar("loss", self.model.loss)
        tf.summary.scalar("accuracy", self.model.acc)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(tensorboard_dir)

        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=session,save_path=self.save_path)

        batch_size = 8
        data_len = len(x_train)
        num_batch = int((data_len - 1) / batch_size) + 1


        # x_val, y_val = process_file(self.val_dir, self.word_to_id, self.cat_to_id, self.config.seq_length)

        loss_in, acc_in = self.evaluate(session, x_val, y_val)
        print ("val loss"+str(loss_in))
        print ("val acc"+str(acc_in))

        print ("start to retrain")
        total_batch = 0  # 总批次
        # best_acc_val = 0.0  # 最佳验证集准确率
        last_improved = 0  # 记录上一次提升批次
        require_improvement = 20  # 如果超过1000轮未提升，提前结束训练

        flag = False
        for epoch in range(self.config.num_epochs):
            print('Epoch:', epoch + 1)
            batch_train = batch_iter(x_train, y_train, self.config.batch_size)
            for x_batch, y_batch in batch_train:
                feed_dict = self.feed_data(x_batch, y_batch, self.config.dropout_keep_prob)

                if total_batch % self.config.save_per_batch == 0:
                    # 每多少轮次将训练结果写入tensorboard scalar
                    s = session.run(merged_summary, feed_dict=feed_dict)
                    writer.add_summary(s, total_batch)

                if total_batch % self.config.print_per_batch == 0:
                    # 每多少轮次输出在训练集和验证集上的性能
                    feed_dict[self.model.keep_prob] = 1.0
                    loss_train, acc_train = session.run([self.model.loss, self.model.acc], feed_dict=feed_dict)
                    loss_val, acc_val = self.evaluate(session, x_val, y_val)  # todo

                    if acc_val > best_acc_val:
                        # 保存最好结果
                        best_acc_val = acc_val
                        last_improved = total_batch
                        saver.save(sess=session, save_path=self.save_path)
                        improved_str = '*'
                    else:
                        improved_str = ''

                    # time_dif = self.get_time_dif(start_time)
                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                          + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, {5}'
                    print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, improved_str))

                session.run(self.model.optim, feed_dict=feed_dict)  # 运行优化
                total_batch += 1

                if total_batch - last_improved > require_improvement:
                    # 验证集正确率长期不提升，提前结束训练
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break  # 跳出循环
            if flag:  # 同上
                break
        return best_acc_val
        # batch_train = batch_iter(x_train, y_train, self.config.batch_size)
        # for x_batch, y_batch in batch_train:
        #     feed_dict = self.feed_data(x_batch, y_batch, self.config.dropout_keep_prob)
        #     feed_dict[self.model.keep_prob] = 1.0
        #     loss_train, acc_train = session.run([self.model.loss, self.model.acc], feed_dict=feed_dict)
        #     loss_val, acc_val = self.evaluate(session, x_val, y_val)
        #
        #     if acc_val > best_acc_val:
        #         # 保存最好结果
        #         best_acc_val = acc_val
        #         # last_improved = total_batch
        #         saver.save(sess=session, save_path=self.save_path)
        #         improved_str = '*'
        #     else:
        #         improved_str = ''
        #
        #     msg = ' Train Loss: {0:>6.2}, Train Acc: {1:>7.2%},' \
        #                       + ' Val Loss: {2:>6.2}, Val Acc: {3:>7.2%} {4}'
        #     print(msg.format( loss_train, acc_train, loss_val, acc_val, improved_str))
        #
        # return best_acc_val


    def score(self, tst_ds):
        print("Loading test data...")
        start_time = time.time()
        # x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)
        x_test, y_test = tst_ds.format_sklearn()
        y_test = kr.utils.to_categorical(y_test, num_classes=3)

        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=session, save_path=self.save_path)  # 读取保存的模型

        # print('Testing...')
        loss_test, acc_test = self.evaluate(session, x_test, y_test)
        # msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
        return acc_test

        #accuracy_score(y, self.predict(X), sample_weight=sample_weight)
        # return accuracy_score(*(test_dataset.format_sklearn()))


    def predict_pro(self, askdataset):
        # rnn_model = RnnModel()
        covlist = []

        unlabeled_entry_ids, X_pool = zip(*askdataset.get_unlabeled_entries())
        for i in X_pool:
            label, pro = self.predict(i)
            covlist.append(np.cov(pro))
            print (label)
            print (pro)
        covlist = np.array(covlist)
        return covlist


    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=self.save_path)  # 读取保存的模型

        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }
        self.categories = ['simple','complicated','preference']
        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        y_pro = self.session.run(self.model.pred_pro, feed_dict = feed_dict)
        # print (y_pro)
        return self.categories[y_pred_cls[0]], y_pro

    def test(self,tst_dst):
        print("Loading test data...")
        start_time = time.time()
        # x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)
        x_test, y_test = tst_dst.format_sklearn()
        y_test = kr.utils.to_categorical(y_test, num_classes=3)

        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=session, save_path=self.save_path)  # 读取保存的模型

        print('Testing...')
        loss_test, acc_test = self.evaluate(session, x_test, y_test)
        msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
        print(msg.format(loss_test, acc_test))

        batch_size = 8
        data_len = len(x_test)
        num_batch = int((data_len - 1) / batch_size) + 1

        y_test_cls = np.argmax(y_test, 1)
        y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
        for i in range(num_batch):  # 逐批次处理
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            feed_dict = {
                self.model.input_x: x_test[start_id:end_id],
                self.model.keep_prob: 1.0
            }
            y_pred_cls[start_id:end_id] = session.run(self.model.y_pred_cls, feed_dict=feed_dict)

        # 评估
        print("Precision, Recall and F1-Score...")
        categories = ['simple', 'complicated', 'preference']
        print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

        # 混淆矩阵
        print("Confusion Matrix...")
        cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
        print(cm)

        time_dif = self.get_time_dif(start_time)
        print("Time usage:", time_dif)

        return acc_test







        
