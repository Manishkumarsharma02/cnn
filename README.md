import tensorflow as tf
from tensorflow import keras
from  tensorflow.keras import datasets,layers,models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
(x_train,y_train),(x_test,y_test)=datasets.cifar10.load_data()
x_train,x_test=x_train/255.0,x_test/255.0
classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck" ]

y_train=y_train.reshape(-1,)
y_test=y_test.reshape(-1,)

def plot_sample(x,y,index):
 plt.figure(figsize=(15,2))
 plt.imshow(x[index])
 plt.xlabel(classes[y[index]])
 plt.show()

for i in range(10):
   plot_sample(x_train,y_train,i)

filters = keras.Sequential([
    #cnn
    layers.Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    #denser
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

filters.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
filters.fit(x_train,y_train,epochs=5)

for i in range (0,9):
 plot_sample(x_test,y_test,i)


y_pred=filters.predict(x_test)
y_pred_classes=[np.argmax(element) for element in y_pred]
y_pred_classes[:5]

print("classfication  report :\n",classification_report(y_test,y_pred_classes)) 


