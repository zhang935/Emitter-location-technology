import codecs
import numpy as np
from tensorflow.keras import backend as K
# K.set_image_data_format('channels_first')

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.utils.vis_utils import plot_model

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow import keras

import numpy as np
# import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import random

import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'
import random
# lr=0.0005 0.67
OPTIMIZER = Adam(lr=0.00002)
class LeNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        # CONV => RELU => POOL
        model.add(Conv2D(64, kernel_size=5, padding="same",
                         input_shape=input_shape))
        model.add(Activation("relu"))
        #model.add(Dropout(0.3))
        model.add(MaxPooling2D(pool_size=(1, 1), strides=(1, 1)))
        # CONV => RELU => POOL
        model.add(Conv2D(64, kernel_size=5, padding="same"))
        model.add(Activation("relu"))
        model.add(Dropout(0.3))
        model.add(Conv2D(128, kernel_size=5, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(1, 1), strides=(1,1)))
        # Flatten => RELU layers
        model.add(Flatten())
        model.add(Dropout(0.3))
        model.add(Dense(60))
        model.add(Activation("relu"))
        model.add(Dropout(0.3))
        # a softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
f = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B1.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f1 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B2.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f2 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B3.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f3 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B4.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f4 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B5.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f5 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B6.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f6 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B7.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f7 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B8.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f8 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B9.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f9 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B10.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f10 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B11.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f11 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B12.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f12 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B13.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f13 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B14.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f14 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B15.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f15 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B16.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f16 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B17.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f17 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B18.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f18 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B19.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f19 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B20.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f20 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B21.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f21 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B22.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f22 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B23.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f23 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B24.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f24 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B25.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f25 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B26.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f26 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B27.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f27 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B28.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f28 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B29.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f29 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B30.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f30 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B31.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f31 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B32.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f32 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B33.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f33 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B34.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f34 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B35.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f35 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B36.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f36 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B37.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f37 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B38.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f38 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B39.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f39 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B40.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f40 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B41.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f41 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B42.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f42 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B43.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f43 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B44.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f44 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B45.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f45 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B46.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f46 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B47.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f47 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B48.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f48 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B49.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f49 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B50.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f50 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B51.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f51 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B52.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f52 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B53.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f53 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B54.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f54 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B55.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f55 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B56.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f56 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B57.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f57 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B58.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f58 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B59.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f59 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B60.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f60 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B61.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f61 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B62.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f62 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B63.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f63 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B64.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f64 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B65.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f65 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B66.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f66 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B67.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f67 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B68.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f68 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B69.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f69 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B70.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f70 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B71.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f71 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B72.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f72 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B73.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f73 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B74.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f74 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B75.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f75 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B76.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f76 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B77.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f77 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B78.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f78 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B79.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f79 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B80.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f80 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B81.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f81 = codecs.open('D:/matlab_work/文渊楼/Blei/DATA/B82.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
def aa(ff,ii):
    line = ff.readline()
    list=[]
    while line:
      a = line.split()
      b = a[3:4]
      list.append(b) # 将其添加在列表之中
      line = ff.readline()
    f.close()
    c=int(len(list)/45)
    list = np.array(list[:c*45]).astype(float)
    list = list.reshape(c, 45).tolist()
    y = [ii for i in range(len(list))]
    return list,y
def bb(ff):
    line = ff.readline()
    list=[]
    while line:
      a = line.split()
      b = a[3:4]
      list.append(b) # 将其添加在列表之中
      line = ff.readline()
    f.close()
    c=int(len(list)/45)
    list = np.array(list[:c*45]).astype(float)
    list = list.reshape(c, 45).tolist()
    return list

list0,y0=aa(f,0)
list1,y1=aa(f1,1)
list2,y2=aa(f2,2)
list3,y3=aa(f3,3)
list4,y4=aa(f4,4)
list5,y5=aa(f5,5)
list6,y6=aa(f6,6)
list7,y7=aa(f7,7)
list8,y8=aa(f8,8)
list9,y9=aa(f9,9)
list10,y10=aa(f10,10)
list11,y11=aa(f11,11)
list12,y12=aa(f12,12)
list13,y13=aa(f13,13)
list14,y14=aa(f14,14)
list15,y15=aa(f15,15)
list16,y16=aa(f16,16)
list17,y17=aa(f17,17)
list18,y18=aa(f18,18)
list19,y19=aa(f19,19)
list20,y20=aa(f20,20)
list21,y21=aa(f21,21)
list22,y22=aa(f22,22)
list23,y23=aa(f23,23)
list24,y24=aa(f24,24)
list25,y25=aa(f25,25)
list26,y26=aa(f26,26)
list27,y27=aa(f27,27)
list28,y28=aa(f28,28)
list29,y29=aa(f29,29)
list30,y30=aa(f30,30)
list31,y31=aa(f31,31)
list32,y32=aa(f32,32)
list33,y33=aa(f33,33)
list34,y34=aa(f34,34)
list35,y35=aa(f35,35)
list36,y36=aa(f36,36)
list37,y37=aa(f37,37)
list38,y38=aa(f38,38)
list39,y39=aa(f39,39)
list40,y40=aa(f40,40)
list41,y41=aa(f41,41)
list42,y42=aa(f42,42)
list43,y43=aa(f43,43)
list44,y44=aa(f44,44)
list45,y45=aa(f45,45)
list46,y46=aa(f46,46)
list47,y47=aa(f47,47)
list48,y48=aa(f48,48)
list49,y49=aa(f49,49)
list50,y50=aa(f50,50)
list51,y51=aa(f51,51)
list52,y52=aa(f52,52)
list53,y53=aa(f53,53)
list54,y54=aa(f54,54)
list55,y55=aa(f55,55)
list56,y56=aa(f56,56)
list57,y57=aa(f57,57)
list58,y58=aa(f58,58)
list59,y59=aa(f59,59)
list60,y60=aa(f60,60)
list61,y61=aa(f61,61)
list62,y62=aa(f62,62)
list63,y63=aa(f63,63)
list64,y64=aa(f64,64)
list65,y65=aa(f65,65)
list66,y66=aa(f66,66)
list67,y67=aa(f67,67)
list68,y68=aa(f68,68)
list69,y69=aa(f69,69)
list70,y70=aa(f70,70)
list71,y71=aa(f71,71)
list72,y72=aa(f72,72)
list73,y73=aa(f73,73)
list74,y74=aa(f74,74)
list75,y75=aa(f75,75)
list76,y76=aa(f76,76)
list77,y77=aa(f77,77)
list78,y78=aa(f78,78)
list79,y79=aa(f79,79)
list80,y80=aa(f80,80)
list81,y81=aa(f81,81)
# print(np.array(list0).shape)

X=list0+list1+list2+list3+list4+list5+list6+list7+list8+list9+list10+list11+list12+list13+list14+list15+list16+list17+list18+list19+list20+list21+list22+list23\
    +list24+list25+list26+list27+list28+list29+list30+list31+list32+list33+list34+list35+list36+list37+list38+list39+list40+list41+list42+list43+list44+list45+list46\
    +list47+list48+list49+list50+list51+list52+list53+list54+list55+list56+list57+list58+list59+list60+list61+list62+list63+list64+list65+list66+list67+list68+list69\
    +list70+list71+list72+list73+list74+list75+list76+list77+list78+list79+list80+list81
print(np.array(X).shape)

Y=y0+y1+y2+y3+y4+y5+y6+y7+y8+y9+y10+y11+y12+y13+y14+y15+y16+y17+y18+y19+y20+y21+y22+y23\
    +y24+y25+y26+y27+y28+y29+y30+y31+y32+y33+y34+y35+y36+y37+y38+y39+y40+y41+y42+y43+y44+y45+y46\
    +y47+y48+y49+y50+y51+y52+y53+y54+y55+y56+y57+y58+y59+y60+y61+y62+y63+y64+y65+y66+y67+y68+y69\
    +y70+y71+y72+y73+y74+y75+y76+y77+y78+y79+y80+y81
cc = list(zip(X,Y))

import random
random.shuffle(cc)
X[:], Y[:] = zip(*cc)
X_train=np.array(X[:8000])
# print(X_train.shape)
X_train=X_train.reshape(8000,15,3)
X_train=X_train[:,np.newaxis,:,:]
y_train=np.array(Y[:8000])
X_test=np.array(X[8000:])
print(X_test.shape)
X_test=X_test.reshape(2796,15,3)
X_test=X_test[:,np.newaxis,:,:]
y_test=np.array(Y[8000:])

model = LeNet.build(input_shape=(1,15,3), classes=82)
model.summary()
model.compile(loss="sparse_categorical_crossentropy", optimizer=OPTIMIZER,
              metrics=["accuracy"])


history = model.fit(X_train, y_train,
                    batch_size=8, epochs=200,
                    verbose=1, validation_split=0.3)

score = model.evaluate(X_test, y_test, verbose=1)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])
model.save('train.h5')
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('acc2.png')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss2.png')
plt.show()

#--------------------------------------------混淆矩阵------------------------------
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# test_model = load_model('class.h5')
y_pred=np.argmax(model.predict(X_test), axis=-1)


labels =  ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25',
          '26','27','28','29','30','31','32','33','34','35','36','37', '38','39','40','41','42','43','44','45','46','47','48','49','50',
          '51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81']
print(len(labels))
tick_marks = np.array(range(len(labels))) + 0.5

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm_normalized)
plt.figure(figsize=(12, 8), dpi=120)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.01:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
# show confusion matrix
# plt.savefig('../Data/confusion_matrix.png', format='png')
plt.savefig('juzhen82.png')
plt.show()