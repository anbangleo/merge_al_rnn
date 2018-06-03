# coding=utf-8
import os
import sys
inp = 'yinanlabels.txt'
out_label = 'labeled.txt'
out_unlabel = 'unlabeled.txt'

f = open(inp,'r')
ol = open(out_label,'w')
ou  = open(out_unlabel,'w')

for line in f.readlines():
	if line.split('\t')[0] != '?':
		num = int(line.split('\t')[0])
#		if num == 0 or num  == 1 or num == 2:
#			ol.writelines('无关 '+line.split('\t')[2])

		if num ==1 or num ==2 or num ==0 or num == 3 or num == 8:
		    ol.writelines('简单'+'\t'+line.split('\t')[2])

		elif num == 4 or num == 5 or num == 9 or num == 10:
		    ol.writelines('复杂'+'\t'+line.split('\t')[2])

		elif num == 6 or num == 7:
		    ol.writelines('倾向'+'\t'+line.split('\t')[2])
		else:
		    print num
	else:
		ou.writelines('未标'+'\t'+line.split('\t')[2]) 
	
f.close()
ol.close()
ou.close()
