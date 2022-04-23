from sklearn import preprocessing
import numpy as np
import codecs

f = codecs.open('D:/matlab_work/文渊楼/C/C1.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
line = f.readline()
list=[]
while line:
    a = line.split()
    b = a[3:4]
    list.append(b) # 将其添加在列表之中
    line = f.readline()
f.close()
list=np.array(list)
min_max_scaler = preprocessing.MinMaxScaler()
X_minMax = min_max_scaler.fit_transform(list)
# list_max=max(list)
# list_min=min(list)
print(X_minMax)
# for i in list:



