# origin = open('queryresult3.txt','r')
import result as rl
import numpy as np
import matplotlib.pyplot as plt
# fi = open('result.txt','r')
# result = open('resultlist.txt','w')

# for line in origin.readlines():
# 	line = line.strip()
# 	if ']' in line:
# 		fi.writelines(line+'\n')
# 	else:
# 		fi.writelines(line+' ')
time1 = []
time16 = []
time64 = []

for i in rl.time1:
    time = i.split('.')[-1].split('\'')[0]
    time1.append(int(time))
for i in rl.time16:
    time = i.split('.')[-1].split('\'')[0]
    time16.append(int(time))
for i in rl.time64:
    time = i.split('.')[-1].split('\'')[0]
    time64.append(int(time))
flag=0
result1 = []
result16 = []
result64 = []
for i in time1:
    if flag % 64 == 0:
        result1.append(i)
        print flag / 64
        print i
    else:
        pass
    flag = flag + 1

flag = 0
for j in time16:
    if flag % 4 == 0:
        result16.append(j)
    else:
        pass
    flag = flag + 1
result64 = time64
query_num = np.arange(1,(len(time64)+1))
plt.plot(query_num, result64, 'r', label='Batch64')
plt.plot(query_num, result16, 'g', label='Batch16')
plt.plot(query_num, result1, 'k', label='Batch1')
plt.ylabel('Time-consuming(ms)')
plt.title('Result')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
           fancybox=True, shadow=True, ncol=5)
plt.show()
