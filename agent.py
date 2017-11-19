
dir = "/home/sam/PycharmProjects/classify/datasets/"

from os import listdir
from PIL import Image
import numpy as np
from sklearn.utils import shuffle
import tflearn
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.estimator import regression

global train_X, train_Y
train_X = []
train_Y = []
test_X = []
test_Y = []
lab_name = ["cat", "dog"] # [1,0] , [0,1]
img_rows = 200
img_col = 200
img_chnl = 3
strides = 3
#
# def read(name):
#     images = []
#     print("Reading " + name + "...")
#     i = 0
#     for file in listdir(dir + name):
#         if not file.startswith('.') and file != 'Thumbs.db':
#             i = i + 1
#             print(i, ":" + file)
#             images = Image.open("datasets/" + name + "/" + file)
#             images = np.asarray(images)
#             train_X.append(images)
#             if name == "cat":
#                 train_Y.append([1, 0])
#             else:
#                 train_Y.append([0, 1])
#
# read("cat")
# read("dog")
# train_X, train_Y = shuffle(train_X, train_Y, random_state=0)

# train_X = np.load(dir + "x.npy")
# train_Y = np.load(dir + "y.n
# To read the data from datasets folder
def read(name):
    images = []
    print("Reading " + name + "...")
    i = 0
    for file in listdir("datasets/" + name):
        if not file.startswith('.') and file != 'Thumbs.db':
            i = i + 1
            print(i, ":" + file)
            images = Image.open("datasets/" + name + "/" + file)
            images = np.asarray(images)
            data.append(images)
            if name == "cat":
                label.append([1, 0])
            else:
                label.append([0, 1])
# To train the model from the data which has already been read
def learn():
    # model learn
    read("cat")
    read("dog")
    model.fit(data, label,
              n_epoch=10, show_metric=True,
              snapshot_step=20, batch_size=100, run_id='cat_dog')
    model.save("cat_dog")
# To Test the trained model

def predict():
    model.load('cat_dog')
    with open('submission_file.csv', 'w') as f:
        f.write('id,label\n')
    with open('submission_file.csv', 'a') as f:
        for file in listdir("datasets/test"):
            image = Image.open("datasets/test/" + file)
            image = np.asarray(image)
            image = image.reshape(-1, 200, 200, 3)
            # p = np.argmax(model.predict(image))
            # print(lab_name[p])
            mod = model.predict(image)[0][1]
            f.write('{},{}\n'.format(file.split(".")[0], mod))
            print(file + ":", mod)
learn()
predict()
# test_X = train_X[:30]
# test_Y = train_Y[:30]

convnet = input_data(shape=[None, 200, 200, 3], name='input')

convnet = conv_2d(convnet, 32, 2, strides=[1, 2, 2, 1], activation='relu')
convnet = max_pool_2d(convnet, [1, 2, 2, 1])

convnet = conv_2d(convnet, 64, 2, strides=[1, 2, 2, 1], activation='relu')
convnet = max_pool_2d(convnet, [1, 2, 2, 1])

convnet = conv_2d(convnet, 64, 2, strides=[1, 2, 2, 1], activation='relu')
convnet = max_pool_2d(convnet, [1, 2, 2, 1])

convnet = conv_2d(convnet, 32, 2, strides=[1, 2, 2, 1], activation='relu')
convnet = max_pool_2d(convnet, [1, 2, 2, 1])

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, learning_rate=0.001, loss='categorical_crossentropy', optimizer='adam')

model = tflearn.DNN(convnet, tensorboard_dir="./Graph", tensorboard_verbose=3)

# model.fit(train_X, train_Y, n_epoch=50, show_metric=True,  batch_size=100, run_id='cat_dog', validation_set=[test_X, test_Y])
#
# model.save(dir+"model/cat_dog")

model.load(dir+"model/cat_dog")

for file in listdir(dir+"test"):
            image = Image.open(dir+"test/" + file)
            image = np.asarray(image)
            image = image.reshape(-1, 200, 200, 3)
            mod = model.predict(image)[0]
            print(str(file)+" : "+lab_name[np.argmax(mod)])




