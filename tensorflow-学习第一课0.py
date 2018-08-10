import tensorflow as tf
import matplotlib.pyplot as plt


node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)# also tf.float32 implicitly
print(node1, node2)
print('------------------------------------------')



#Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0",shape=(), dtype=float32)


sess = tf.Session()
print(sess.run([node1, node2]))


node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))
print('-------------------------------------------------')


'''
node3:Tensor("Add:0", shape=(), dtype=float32)
sess.run(node3):7.0

'''
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

print(sess.run(adder_node, {a:3, b:4.5}))
print(sess.run(adder_node, {a: [1,3], b: [2,4]}))

print('-------------------------------------------------')

'''
7.5
[3.  7.]

'''
add_and_triple = adder_node *3.
print(sess.run(add_and_triple, {a:3, b:4.5}))
print('----------------------------------------')

'''

输出结果是：
22.5

'''

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b

init = tf.global_variables_initializer()
sess.run(init)


print(sess.run(linear_model, {x: [1,2,3,4]}))
print('--------------------------------------------')
'''
求值linear_model 
输出为
[0.  0.30000001  0.60000002  0.90000004]


'''

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1,2,3,4], y: [0, -1, -2, -3]}))

print('---------------------------------------')
'''
输出的结果为
23.66
'''

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1,2,3,4], y: [0, -1, -2, -3]}))
print('---------------------------------------------')
'''
最终打印的结果是：
0.0
'''
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter([1,2,3,4], [0, -1, -2, -3])
#plt.ion()
plt.show(block=False)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)# reset values to incorrect defaults.
for i in range(1000):
     sess.run(train, {x: [1,2,3,4], y: [0, -1, -2, -3]})
     if i % 50 ==0:
          try:         
            ax.lines.remove(lines[0])
          except Exception:
            pass
          prediction_y=sess.run(linear_model, {x: [1,2,3,4], y: [0, -1, -2, -3]})
          lines=ax.plot([1,2,3,4],prediction_y,'r-',lw=5)
          plt.pause(0.5)              

 
          print(i,sess.run([W,b]))
print('----------------------------------------------')

'''输出结果为
[array([-0.9999969], dtype=float32), array([ 0.99999082], dtype=float32)]
'''
