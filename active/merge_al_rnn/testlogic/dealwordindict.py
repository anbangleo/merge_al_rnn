# coding: utf-8

import sys
from collections import Counter

import numpy as np
import tensorflow.contrib.keras as kr
from sklearn import preprocessing
import jieba

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False

# class Dealword():
def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if not is_py3:
        return word.encode(encoding)
    else:
        return word


def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content


def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)

def particularword(content):
    result = []
    strength = len(content)
    result.append(float(strength))

    words = list(jieba.cut(native_content(content)))

    allwords = len(words)
    wenhao = 0
    tanhao = 0
    weus = 0
    you = 0
    for i in words:
        if i == '?' or i =='？':
            wenhao = wenhao + 1
        elif i == '!' or i=='！':
            tanhao = tanhao + 1
        elif i == '我' or i == '我们' or i == '咱' or i =='咱们':
            weus = weus + 1
        elif i == '你' or i == '你们':
            you = you + 1
        else:
            pass
    result.append(float(wenhao))
    result.append(float(tanhao))
    result.append(float(weus/allwords))
    result.append(float(you/allwords))
    #result.append(format(float(weus)/float(allwords),'.6f'))
    #result.append(format(float(you)/float(allwords),'.6f')) 

    particu = [
    ["欧拉", "公式", "定理", "证明", "推理", "归纳", "面体", "方程", "数列", "奇点", "三角形",   "猜想", "化归", "斐波那契", "几何", "奇数", "概括", "结论", "图形", "演绎", "拓扑", "分析"]
,["作业", "考试", "成绩", "交", "教辅", "发布", "测试", "扣", "分", "附件", "下载", "平台", "答疑", "讨论", "网络", "视频"]
,["吗", "希望", "教", "什么", "呢", "几", "请问", "啥", "请", "哪", "谁", "大家", "帮", "求", "神马", "怎么", "有人"],
["怎样", "是否", "如何", "为什么", "还是", "为何", "会不会", "能不能",
    "难道", "别的", "可不可以", "有没有", "哪些", "是不是", "还有"]
,["应该", "是", "记得", "应当", "可能", "就是"],
["说明", "因为", "认为", "可以", "所以", "通过", "如果", "那么", "想法", "觉得", "另外", "否则", "而且", "于是", "不是", "而是", "个人", "由于", "可知", "必定", "即可", "假设"]
,["例如", "比如", "例子"],
["一样", "就像", "好像", "看成", "转化", "联想", "相似", "想起", "类似", "想到"]
,["不够", "但", "更", "并非", "并不", "但是", "可是", "不", "不是很", "不同意", "不赞",
    "不喜欢", "没意思", "不感兴趣", "不是的", "不正确", "不对", "没道理", "没帮助"],
["感谢", "喜欢", "有意思", "感兴趣", "有趣", "很好", "好/a", "不错", "是的", "赞", "的确",
    "同意", "没错", "正确", "对", "道理", "也", "谢谢", "很有", "有帮助", "收获", "同感", "生动", "的确"],
[ "呵", "呵呵","哈哈", "谢", "感觉", "太", "懂", "难", "赞", "辛苦", "吃力", "很", "好"]
]
    temp = 0
    for p in particu:
        for q in p:
            for i in words:
                if i==q:
                    temp = temp + 1
                else:
                    pass
        result.append(float(temp/allwords))
        #result.append(format(float(temp)/float(allwords),'.6f'))
        temp = 0 
#    print (result)
    return result



def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            #try:
            label, content = line.strip().split('\t')
            if content:
                contentd = particularword(content)
                contents.append(contentd)
                labels.append(native_content(label))
#            except:
 #               pass
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=1000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)
    print ("start to build vob....")
    all_data = []
    # print (data_train) 
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')
    print ("Finish building vocab!")

def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    """读取分类目录，固定"""
    categories = ['简单','复杂', '倾向']

    categories = [native_content(x) for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(filename, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    print ("Start to deal with file...")
    contents, labels = read_file(filename)
    data_id, label_id = [], []
    for i in range(len(contents)):
        label_id.append(cat_to_id[labels[i]])

    x_pad = np.array(contents)
    #均一化
   
 #x_pad = preprocessing.scale(x_pad)

    # 使用keras提供的pad_sequences来将文本pad为固定长度
   # x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示
    print ("Finish dealing with files!")
    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
