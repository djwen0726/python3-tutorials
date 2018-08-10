import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#使用np生产100个随机点

x_data= np.random.rand(100)
noise = np.random.normal(0,0.02,x_data.shape)
y_data=x_data*0.3+0.1+noise
#构造一个线性模型

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
#plt.ion()
plt.show(block=False)

b = tf.Variable(0.0)
k= tf.Variable(0.0)
y=k*x_data+b

#定义二次代价函数
loss = tf.reduce_mean(tf.square(y_data-y))
#定义一个梯度下降法来进行训练优化器
optimizer = tf.train.ProximalGradientDescentOptimizer(0.5)
#最小化代价函数
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step%20 ==0:
          try:         
            ax.lines.remove(lines[0])
          except Exception:
               pass
          prediction_y = sess.run(y)
          lines=ax.plot(x_data,prediction_y,'r-',lw=5)
          
          plt.pause(0.5)        

             
          print(step,sess.run([k,b,loss]))

  
