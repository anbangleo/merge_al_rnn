# coding: utf-8

from __future__ import print_function

import os
import tensorflow as tf
import tensorflow.contrib.keras as kr

from rnn_model import TRNNConfig, TextRNN
from data.cnews_loader import read_category, read_vocab
import numpy as np

try:
    bool(type(unicode))
except NameError:
    unicode = str
base_dir = '/home/ab/test/al/active/data/yinan/rnn'
vocab_dir = os.path.join(base_dir, 'cnews.vocab1.txt')

save_dir = 'checkpoints/textrnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


class RnnModel:
    def __init__(self):
        self.config = TRNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextRNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }
        self.categories = ['simple','complicated','preference'] 
        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        y_pro = self.session.run(self.model.pred_pro, feed_dict = feed_dict)
        print (y_pro)
        # print (type(y_pro))
        print (np.cov(y_pro))
        return self.categories[y_pred_cls[0]]


if __name__ == '__main__':
    rnn_model = RnnModel()

    test_demo = ['哈哈哈，老师说长久不用的拖把上会长出蘑菇，然而我一次都没看到我家拖把上有蘑菇23333','相比传统绘画来说，我个人更喜欢印象派作品，这种随意但不失风采的绘画方式值得我们学习。'
,'酚酞溶液变红就一定能确认是氢氧化钠吗？'
]
    for i in test_demo:
        print ( rnn_model.predict(i))
#        print (unicode(rnn_model.predict(i),encoding="utf-8"))
