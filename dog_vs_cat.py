import numpy as np
import cv2
import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split

cat_imgs = []
dog_imgs = []

i = 0
for c, d in zip(glob.glob('ここにKaggleのデータのパス（cat）を記述')[:100], \
    glob.glob('ここにKaggleのデータのパス（dog）を記述')[:100]):
    cat_imgs.append(cv2.resize(cv2.imread(c), dsize=(224,224)))
    dog_imgs.append(cv2.resize(cv2.imread(d), dsize=(224,224)))
    i += 1
    print('\r' + str(i), end='')

imgs = cat_imgs + dog_imgs
imgs = np.array(imgs)

labels = np.concatenate([np.zeros(len(cat_imgs)),np.ones(len(dog_imgs))])
train_imgs, test_imgs, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.3, random_state=0)

# ニューラルネットワーク

model1 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(224,224,3)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

# print(model1.summary())
model1.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model1.fit(train_imgs,train_labels,epochs=10)

test1_result = model1.evaluate(test_imgs,test_labels)
print('normal nn accuracy' + str(test1_result))

# CNNを使用

model2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(2,2),   
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(256,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(512,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

# print(model2.summary())
model2.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model2.fit(train_imgs,train_labels,epochs=10)
print('cnn accuracy' + str(model2.evaluate(test_imgs,test_labels)))

# 転移学習

base_model = tf.keras.applications.vgg16.VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))
base_model.trainable = False

model_teni = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

# print(model_teni.summary())
model_teni.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model_teni.fit(train_imgs,train_labels,epochs=5)
print('teni accuracy' + str(model_teni.evaluate(test_imgs,test_labels)))

def judge(photo_name):
    photo = cv2.resize(cv2.imread(photo_name), dsize=(224,224))
    photo.resize(1,224,224,3)
    judi_num = model_teni.predict(photo)
    if judi_num >= 0.5:
        test = 'dog'
    else:
        test = 'cat'
    return test

print(judge('ここに調べたい画像のパスを記述'))