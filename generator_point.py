import numpy as np
import random
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn import preprocessing
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
# np.random.seed(0)


class Generator_Point:

    def transform_one_hot(self, x):
        classes = int(max(x))+1
        one_hot_label = np.zeros(shape=(len(x), classes))
        one_hot_label[np.arange(0, len(x)), x] = 1
        return one_hot_label

    def simple_two_class(self):
        X, y = make_blobs(100, 2, centers=2, random_state=2)
        #plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap='RdBu')
        #plt.show()
        label = self.transform_one_hot(y)
        del y
        return X, label

    def not_onehot(self):
        X, y = make_blobs(100, 2, centers=2, random_state=2)
        return X,y



class Generate_csv:
    def read_csv(self):
        pima = pd.read_csv('pima-indians-diabetes.data.csv').values
        X_scaled = preprocessing.MinMaxScaler()
        data = pima[:, :8]
        data = X_scaled.fit_transform(data)
        labels = self.transform_one_hot(pima[:, 8].astype(np.int32))
        return data, labels

    def iris_csv(self):
        data = load_iris().data
        X_scaled = preprocessing.MinMaxScaler()
        data = X_scaled.fit_transform(data)
        labels = load_iris().target
        labels = self.transform_one_hot(labels)
        return data, labels


    def read_not_onehot(self):
        pima = pd.read_csv('pima-indians-diabetes.data.csv').values
        data = pima[:, :8]
        X_scaled = preprocessing.MinMaxScaler()
        data = X_scaled.fit_transform(data)
        labels = pima[:, 8].astype(np.int32)
        return data, labels

    def transform_one_hot(self, x):
        classes = int(max(x))+1
        one_hot_label = np.zeros(shape=(len(x), classes))
        one_hot_label[np.arange(0, len(x)), x] = 1
        return one_hot_label

Generate_csv().iris_csv()

def get_batches(data, label):
    #i = random.randint(0, len(data)-1)
    for i in range(len(data)):
        yield data[i:i+1], label[i:i+1]

'''
if __name__ != '__main__':
    x = tf.placeholder(tf.float32, [None, 8])
    y = tf.placeholder(tf.float32, [None, 2])


    W1 = tf.Variable((tf.truncated_normal([8,8],stddev=0.1)))
    #W1=tf.get_variable(name='w',shape=[2,2],initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.constant(0.1, shape=(1, 8)))

    W2 = tf.Variable(tf.truncated_normal([8, 2], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=(1, 2)))
    z1 = tf.add(tf.matmul(x, W1), b1)
    o1 = tf.nn.sigmoid(z1)
    #o1=tf.nn.softmax(f1)
    z2 = tf.add(tf.matmul(o1, W2), b2)
    o2 = tf.nn.softmax(z2) 


    acc = tf.equal(tf.arg_max(o2,1), tf.arg_max(y,1))
    accuracy = tf.reduce_mean(tf.cast(acc, tf.float32))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=o2, labels=y))
    train_step = tf.train.GradientDescentOptimizer(1e-1).minimize(loss)
    data, label = Generate_csv().get_samples()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(10000):
            data1, label1 = data.reshape(-1, 8), label.reshape(-1, 2)
            #print(sess.run(o4, feed_dict={x: data1, y: label1}))
            single_acc, single_loss, _ = sess.run([accuracy, loss, train_step], feed_dict={x: data1, y: label1})
            print('epoch:{} accuracy:{}  loss:{}'.format(epoch+1, single_acc, single_loss))
'''