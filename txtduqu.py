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
# 0.00003
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
        model.add(Dense(50))
        model.add(Activation("relu"))
        model.add(Dropout(0.3))
        # a softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
f = codecs.open('D:/matlab_work/文渊楼/shiyan/2/A1.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f1 = codecs.open('D:/matlab_work/文渊楼/shiyan/2/A2.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f2 = codecs.open('D:/matlab_work/文渊楼/shiyan/2/A3.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f3 = codecs.open('D:/matlab_work/文渊楼/shiyan/2/A4.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
f4 = codecs.open('D:/matlab_work/文渊楼/shiyan/2/A5.txt', mode='r', encoding='utf-8') # 打开txt文件，以‘utf-8'编码读取
s1=15*2
def aa(ff,ii):
    line = ff.readline()
    list=[]
    while line:
      a = line.split()
      b = a[3:4]
      list.append(b) # 将其添加在列表之中
      line = ff.readline()
    f.close()
    c=int(len(list)/s1)
    list = np.array(list[:c*s1]).astype(float)
    list = list.reshape(c, s1).tolist()
    y = [ii for i in range(len(list))]
    return list,y

# list0=np.array(list0[:11250]).astype(float).tolist()
list0,y0=aa(f,0)
list1,y1=aa(f1,1)
list2,y2=aa(f2,2)
list3,y3=aa(f3,3)
list4,y4=aa(f4,4)

# lista=list0+list2+list3+list5+list8+list6
# listb=list1+list7+list4
# print(np.array(lista).shape)
# print(np.array(listb).shape)
# lista=np.array(lista[:29430])
# lista=lista.reshape(654,45).tolist()
# listb=np.array(listb[:29430])
# listb=listb.reshape(654,45).tolist()

# lista=list0+list2+list3+list5+list8+list6+list1+list7+list4
# print(np.array(lista).shape)
X=list0+list1+list2+list3+list4
Y=y0+y1+y2+y3+y4
cc = list(zip(X,Y))

import random
random.shuffle(cc)
X[:], Y[:] = zip(*cc)
X_train=np.array(X[:500])
print(np.array(X).shape)
X_train=X_train.reshape(500,15,2)
X_train=X_train[:,np.newaxis,:,:]
y_train=np.array(Y[:500])
X_test=np.array(X[500:])
print(X_test.shape)
X_test=X_test.reshape(247,15,2)
X_test=X_test[:,np.newaxis,:,:]
y_test=np.array(Y[500:])



model = LeNet.build(input_shape=(1,15,2), classes=5)
model.summary()
model.compile(loss="sparse_categorical_crossentropy", optimizer=OPTIMIZER,
              metrics=["accuracy"])


history = model.fit(X_train, y_train,
                    batch_size=4, epochs=50,
                    verbose=1, validation_split=0.1)

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

plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.show()

#--------------------------------------------混淆矩阵------------------------------
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#test_model = model.keras.models.load_model('class.h5')
y_pred=model.predict_classes(X_test)


#labels表示你不同类别的代号，比如这里的demo中有9个类别
labels = ['0','1', '2','3','4']
tick_marks = np.array(range(len(labels))) + 0.5

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.Reds):
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
plt.savefig('juzhen2.png')
plt.show()