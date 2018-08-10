import tensorflow as tf
import numpy as np

#creat data
x_data = np.random.rand(100).astype(np.float32)
print(type(x_data))
y_data = x_data*8.15+3.1415926
print(type(y_data))

print(x_data)
print('---------------------------------------')
print(y_data)


Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))

biases = tf.Variable(tf.zeros([1]))

print(Weights)
print(biases)

y = Weights*x_data+biases

loss= tf.reduce_mean(tf.square(y-y_data))



optimizer = tf.train.GradientDescentOptimizer(0.6)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()


sess = tf.Session()
sess.run(init)

for step in range(301):
     sess.run(train)
     if step % 20 ==0:
          print(step,8.15-sess.run(Weights),3.1415926-sess.run(biases))
