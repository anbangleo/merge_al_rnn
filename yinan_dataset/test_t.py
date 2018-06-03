# coding=utf-8
fi = 'labeled.txt'
f = open(fi,'r')

for line in f.readlines():
    print (line.split('\t'))
