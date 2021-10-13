#!/usr/bin/env python
# coding: utf-8

# # <center>HW5</center> #
# ####  107000105 楊晶宇
# ***
# ## Implementation
# 
# ### 提醒:檔案內用到許多圖片檔，最後predict有用到model檔，因為不知道檔案會不會太大，所以沒上傳，助教需要可以跟我取得，我怕先上傳太佔空間

# ### Data preprocessing<br>
# 直接去keras dataset下載檔案

# In[2]:


import numpy as np 
import pandas as pd 
import tensorflow as tf
import keras

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()


# 為了要符合model input形式，將data reshape，並宣告labels

# In[4]:


X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
labels=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']


# 把train data前20個畫出來觀察一下，看一下label有沒有錯

# In[185]:


plt.figure(figsize = (15, 15))
for i in range(20):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.xlabel(labels[y_train[i]])
    plt.imshow(X_train[i].reshape(28, 28))


# ### train test split<br>
# <br>
# 因為在training的過程原則上是完全不能看到test data的，所以我們把training data  拆出一部分當 val，**目的是觀察model的穩定性**，看是否設計的model有overfiting的現象，**如果有的話，則要去修改model的架構**(增加drop out或 batch normalization)，讓model穩定，**最後再把val+train用穩定的parameter丟進穩定的model training，得出最好的classifier**

# from sklearn.model_selection import train_test_split 
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=0, stratify=y_train)

# ## Model design<br>
# ### 3layers model
# 因為是圖片，我們選擇用CNN，因為CNN有用到**filter convolution**的概念，比較能夠**取出圖片上的特徵**，且和DNN比減少較多參數。
# 層數上我用了三層，原因是因為希望能夠先把training set train到飽和(99%)，再去處理overfitting的問題，**兩層有點太少，training set沒辦法到99%，不能確定model是不是真的到極限了**<br>
# 
# Kernel size都設定3*3，因為圖片本身大小只有28*28，不能設太大，conv2D後面都加BatchNormalization去讓model穩定，不要overfitting，也會換成Dropout**(後面會比較結果)，最後經過三層DNN輸出(因為前面取出的feature有點多，filter有512個)<br>
# 
# model complexity: **parameters數量為1,937,354**

# In[11]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization, Dropout
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
# model.add(BatchNormalization())
model.add(Dropout(rate=0.5))    
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(Dropout(rate=0.5))    
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(BatchNormalization())
model.add(Dropout(rate=0.5))    
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(Dropout(rate=0.5))           
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(BatchNormalization())
model.add(Dropout(rate=0.5))     
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(Dropout(rate=0.5))        
model.add(MaxPool2D(pool_size=(2, 2)))   
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()


# ### 2layers model
# 後來發現2層的model沒有比較差，雖然層數比較少，但是最後穩定停在的ACC比3層的還要高，**後面的分析會分成兩部分，一部份是3layers，一部分是2layers**
# 
# model complexity: **parameters數量為703,762**，大約是3layers的0.36倍

# In[12]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization, Dropout

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape = X_train.shape[1:]))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2), padding='same'))
model.add(Dropout(0.5))
# model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2), padding='same'))
model.add(Dropout(0.5))
# model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(216, activation='relu'))
model.add(Dropout(0.5))
# model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))
model.summary()


# In[6]:


model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape = X_train.shape[1:]))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2), padding='same'))
model.add(Dropout(0.25))
# model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2), padding='same'))
model.add(Dropout(0.25))
# model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(216, activation='relu'))
model.add(Dropout(0.5))
# model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))
model.summary()


# 由於epochs如果設太大，training我自己電腦會跑不動，所以我都用**google colab提供的免費GPU跑**，這樣epoch就可以設大一點

# In[18]:


from keras import optimizers

lr = 0.005
epochs = 250
batch_size = 1024

optimizer = optimizers.Adagrad(lr=lr)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics = ['accuracy'])


# In[ ]:


history = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_val, y_val))


# ### Model Selection and Parameter Selection
#   主要是把loss跟ACC的趨勢圖畫出來，分析哪一個model配parameters在testing set上面表現會比較好，選擇作為最後的momdel跟parameters
#   #### 圖片都存下來，後面一起分析

# In[247]:


import matplotlib.pyplot as plt
plt.title('Model Accuracy lr=' + str(lr) + ' epochs = ' + str(epochs))
plt.plot(range(1, epochs+1), history.history['accuracy'])
plt.plot(range(1, epochs+1), history.history['val_accuracy'])
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.legend(labels=['train','validation'])
plt.show()
plt.savefig('Model_Accuracy_lr_' + str(lr) + '_epochs_' + str(epochs) + '.png')

plt.title('Model Loss lr=' + str(lr) + ' epochs = ' + str(epochs) + '.png')
plt.plot(range(1, epochs+1),history.history['loss'])
plt.plot(range(1, epochs+1),history.history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend(labels=['train','validation'])
plt.show()
plt.savefig('Model_Loss_lr_' + str(lr) + '_epochs_' + str(epochs) + '.png')


# ### 下面會把所有試過的model跟參數整理比較(圖上標題都有紀錄參數)：
# <br>

# In[23]:


import matplotlib.pyplot as plt
img_acc_list = []
img_acc_list.append(plt.imread('pic/5.png'))
img_acc_list.append(plt.imread('pic/8.png'))
img_acc_list.append(plt.imread('pic/11.png'))
img_loss_list = []
img_loss_list.append(plt.imread('pic/6.png'))
img_loss_list.append(plt.imread('pic/9.png'))
img_loss_list.append(plt.imread('pic/12.png'))

plt.figure(figsize = (50, 50))
j = 1
for i in range(3):
    plt.subplot(3, 3, j)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_acc_list[i])
    j = j+1
for i in range(3):
    plt.subplot(3, 3, j)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_loss_list[i])
    j = j + 1


# ### 3layers bactchsize normalization model(上面的figure)<br>
# 1.最左邊一排是第一次試，發現還沒train完整，ACC還在上升，所以把epochs加到200<br>
# 2.中間那排是加到200，發現還是沒train完，所以決定調大learning rate，不然太花時間了<br>
# 3.最右邊的是learning rate調大後的結果，發現val ACC穩定在0.88附近震盪，train set也train到接近100%，但是val loss最後增加了<br>

# In[10]:


import matplotlib.pyplot as plt
img_acc_list = []
img_acc_list.append(plt.imread('pic/15_dropout.png'))
img_loss_list = []
img_loss_list.append(plt.imread('pic/16_dropout.png'))


plt.figure(figsize = (50, 50))
j = 1
for i in range(1):
    plt.subplot(1, 2, j)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_acc_list[i])
    j = j+1
for i in range(1):
    plt.subplot(1, 2, j)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_loss_list[i])
    j = j + 1


# ### 3layers dropout model(上面的figure)
# 測試3layers dropout發現和batch normalization同樣的learning rate，epoch到120還只停在0.71左右，上升的速度比batch normalization 慢很多(**可以對比上一張figure最右邊的圖，epochs在125的時候model已經穩定且到達0.88了**)，感覺要train完要調大更多的epochs，跑的時間太久了，所以就沒有train了，直接把model complexity降低，試2層的model

# In[13]:


import matplotlib.pyplot as plt
img_acc_list = []
img_acc_list.append(plt.imread('pic/17.png'))
img_loss_list = []
img_loss_list.append(plt.imread('pic/18.png'))


plt.figure(figsize = (50, 50))
j = 1
for i in range(1):
    plt.subplot(1, 2, j)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_acc_list[i])
    j = j+1
for i in range(1):
    plt.subplot(1, 2, j)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_loss_list[i])
    j = j + 1


# ### 2layers batch normalization model(上面的figure)

# 感覺validation前面震盪很嚴重，但是後面有逐漸穩定的趨勢，train也有接近99%，但是val的loss最後有在上升的趨勢，先試試看2 layers的dropout

# In[16]:


import matplotlib.pyplot as plt
img_acc_list = []
img_acc_list.append(plt.imread('pic/15_dropout.png'))
img_acc_list.append(plt.imread('pic/19_dropout.png'))
img_loss_list = []
img_loss_list.append(plt.imread('pic/16_dropout.png'))
img_loss_list.append(plt.imread('pic/20_dropout.png'))



plt.figure(figsize = (50, 50))
j = 1
for i in range(2):
    plt.subplot(2, 2, j)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_acc_list[i])
    j = j+1
for i in range(2):
    plt.subplot(2, 2, j)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_loss_list[i])
    j = j + 1


# ### 2layers dropout  model(上面的figure)

# 1.左邊的是一開始測dropout，發現train了125個epochs還沒train完，所以把epochs、learning rate調大<br>
# 2.右邊把epochs調大成250，lr調大成0.005，發現整體穩定下來了無論是loss或acc，且training跟val的差別沒有很大，比較沒有overfitting的問題，決定選這組當成是**最佳training的model**

# ## training all training dataset
# 上一步我們選出的最佳model是 **2layersdropout lr=0.005 epochs=250**<br>
# 確定model穩定，沒有嚴重的overfitting之後，把所有training data(val + train)丟進model再retrain一次(用調好的參數)，讓val的資訊不要浪費掉

# In[214]:


X_train = np.vstack((X_train, X_val))
y_train = np.hstack((y_train, y_val))


# In[215]:


model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

lr = 0.0005
epochs = 75
batch_size = 1024

model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_val, y_val))


# ## final results<br>
# 最後的結果準確率為**0.9211**，順便把confution matrix印出來看一下分布狀況，發現**T-shirt被誤判成T-shirt/top類的比例比較高**，其實是合理的，因為都是T-shirt，差別只在領口的高度，加上解析度又不好，判斷錯比較多情有可原

# In[19]:


from keras.models import load_model
model = keras.models.load_model('final.h5')
from sklearn.metrics import accuracy_score
y_pred = np.argmax(model.predict(X_test), axis=-1)
acc = accuracy_score(y_test, y_pred)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index = labels,
                  columns = labels)
plt.figure(figsize = (10,7))
plt.title('ACC = '+ str(acc) + ' lr=' + str(lr) + ' epochs = ' + str(epochs) + '.png')
sn.heatmap(df_cm, annot=True)


# ## conclusion
# 這次的作業學到滿多的，主要是如何選擇model與條整參數，我發現以下幾點：<br>
# 
# 1.三層的CNN沒有比兩層的好，層數越多不代表ACC就會越高<br>
# 2.dropout跑的速度比batch normalization慢，但是結果不一定比較差，跑得夠久，還是可以穩定下來<br>
# 3.learning rate越小，上升的速度越慢，train完的時間需要得越久<br>
# 
# 整個作業做完更了解CNN架構，也學會怎麼運用免費的GPU去跑實驗，測試比較大的epoch跟batchsize，很有成就感
