import input_data
import tensorflow as tf

mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)#MNIST数据输入
x=tf.placeholder(tf.float32,[None,784])#图像输入向量
W=tf.Variable(tf.zeros([784,10]))#权重，初始化为0
b=tf.Variable(tf.zeros([10]))#BIAS，初始化为0

y=tf.nn.softmax(tf.matmul(x,W)+b)#模型计算，y为预测值，y_为实际值
y_=tf.placeholder("float",[None,10])


cross_entropy=-tf.reduce_sum(y_*tf.log(y))#计算交叉熵
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)#BP算法调整参数

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)#随机抓取训练数据中的100个批处理数据点
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))

print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
