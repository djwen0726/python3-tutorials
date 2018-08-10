'''
介绍
先前的教程展示了一个简单的线性模型，对MNIST数据集中手写数字的识别率达到了91%。
在这个教程中，我们会在TensorFlow中实现一个简单的卷积神经网络，它能达到大约99%的分类准确率，如果你做了一些建议的练习，准确率还可能更高。
卷积神经网络在一张输入图片上移动一个小的滤波器。这意味着在遍历整张图像来识别模式时，要重复使用这些滤波器。这让卷积神经网络在拥有相同数量的变量时比全连接网络（Fully-Connected）更强大，也让卷积神经网络训练得更快。
你应该熟悉基本的线性代数、Python和Jupyter Notebook编辑器。如果你是TensorFlow新手，在本教程之前应该先学习第一篇教程。


流程图
下面的图表直接显示了之后实现的卷积神经网络中数据的传递。
from IPython.display import Image
Image('images/02_network_flowchart.png')复制代码
输入图像在第一层卷积层里使用权重过滤器处理。结果在16张新图里，每张代表了卷积层里一个过滤器（的处理结果）。图像经过降采样，分辨率从28x28减少到14x14。
16张小图在第二个卷积层中处理。这16个通道以及这层输出的每个通道都需要一个过滤权重。总共有36个输出，所以在第二个卷积层有16 x 36 = 576个滤波器。输出图再一次降采样到7x7个像素。
第二个卷积层的输出是36张7x7像素的图像。它们被转换到一个长为7 x 7 x 36 = 1764的向量中去，它作为一个有128个神经元（或元素）的全连接网络的输入。这些又输入到另一个有10个神经元的全连接层中，每个神经元代表一个类别，用来确定图像的类别，即图像上的数字。
卷积滤波一开始是随机挑选的，因此分类也是随机完成的。根据交叉熵（cross-entropy）来测量输入图预测值和真实类别间的错误。然后优化器用链式法则自动地将这个误差在卷积网络中传递，更新滤波权重来提升分类质量。这个过程迭代了几千次，直到分类误差足够低。
这些特定的滤波权重和中间图像是一个优化结果，和你执行代码所看到的可能会有所不同。
注意，这些在TensorFlow上的计算是在一部分图像上执行，而非单独的一张图，这使得计算更有效。也意味着在TensorFlow上实现时，这个流程图实际上会有更多的数据维度。
卷积层
下面的图片展示了在第一个卷积层中处理图像的基本思想。输入图片描绘了数字7，这里显示了它的四张拷贝，我们可以很清晰的看到滤波器是如何在图像的不同位置移动。在滤波器的每个位置上，计算滤波器以及滤波器下方图像像素的点乘，得到输出图像的一个像素。因此，在整张输入图像上移动时，会有一张新的图像生成。
红色的滤波权重表示滤波器对输入图的黑色像素有正响应，蓝色的代表有负响应。
在这个例子中，很明显这个滤波器识别数字7的水平线段，在输出图中可以看到它对线段的强烈响应。
Image('images/02_convolution.png')复制代码
滤波器遍历输入图的移动步长称为stride。在水平和竖直方向各有一个stride。
在下面的源码中，两个方向的stride都设为1，这说明滤波器从输入图像的左上角开始，下一步移动到右边1个像素去。当滤波器到达图像的右边时，它会返回最左边，然后向下移动1个像素。持续这个过程，直到滤波器到达输入图像的右下角，同时，也生成了整张输出图片。
当滤波器到达输入图的右端或底部时，它会用零（白色像素）来填充。因为输出图要和输入图一样大。
此外，卷积层的输出可能会传递给修正线性单元（ReLU），它用来保证输出是正值，将负值置为零。输出还会用最大池化(max-pooling)进行降采样，它使用了2x2的小窗口，只保留像素中的最大值。这让输入图分辨率减小一半，比如从28x28到14x14。
第二个卷积层更加复杂，因为它有16个输入通道。我们想给每个通道一个单独的滤波，因此需要16个。另外，我们想从第二个卷积层得到36个输出，因此总共需要16 x 36 = 576个滤波器。要理解这些如何工作可能有些困难


'''
#导入模块

#%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math


tf.__version__

# 神经网络的配置
#方便起见，在这里定义神经网络的配置，你可以很容易找到或改变这些数值，然后重新运行Notebook。


# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

#载入数据
#MNIST数据集大约12MB，如果没在文件夹中找到就会自动下载。

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

'''
现在已经载入了MNIST数据集，它由70,000张图像和对应的标签（比如图像的类别）组成。
数据集分成三份互相独立的子集。我们在教程中只用训练集和测试集。
'''

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

'''
类型标签使用One-Hot编码，这意外每个标签是长为10的向量，除了一个元素之外，
其他的都为零。这个元素的索引就是类别的数字，即相应图片中画的数字。我们也
需要测试数据集类别数字的整型值，用下面的方法来计算。
'''

data.test.cls = np.argmax(data.test.labels, axis=1)

'''
数据维度
在下面的源码中，有很多地方用到了数据维度。它们只在一个地方定义，因此我们
可以在代码中使用这些数字而不是直接写数字。
'''

# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10

'''
用来绘制图片的帮助函数

这个函数用来在3x3的栅格中画9张图像，然后在每张图像下面写出真实类别和预测类别。
'''

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


#绘制几张图像来看看数据是否正确
# Get the first images from the test-set.
images = data.test.images[0:9]

# Get the true classes for those images.
cls_true = data.test.cls[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)

'''
TensorFlow图
TensorFlow的全部目的就是使用一个称之为计算图（computational graph）的东西，它会比直接在Python中进行相同计算量要高效得多。TensorFlow比Numpy更高效，因为TensorFlow了解整个需要运行的计算图，然而Numpy只知道某个时间点上唯一的数学运算。
TensorFlow也能够自动地计算需要优化的变量的梯度，使得模型有更好的表现。这是由于图是简单数学表达式的结合，因此整个图的梯度可以用链式法则推导出来。
TensorFlow还能利用多核CPU和GPU，Google也为TensorFlow制造了称为TPUs（Tensor Processing Units）的特殊芯片，它比GPU更快。
一个TensorFlow图由下面几个部分组成，后面会详细描述：

占位符变量（Placeholder）用来改变图的输入。
模型变量（Model）将会被优化，使得模型表现得更好。
模型本质上就是一些数学函数，它根据Placeholder和模型的输入变量来计算一些输出。
一个cost度量用来指导变量的优化。
一个优化策略会更新模型的变量。

另外，TensorFlow图也包含了一些调试状态，比如用TensorBoard打印log数据，本教程不涉及这些。
创建新变量的帮助函数
函数用来根据给定大小创建TensorFlow变量，并将它们用随机值初始化。需注意的是在此时并未完成初始化工作，仅仅是在TensorFlow图里定义它们。


'''


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

'''
创建卷积层的帮助函数
这个函数为TensorFlow在计算图里创建了新的卷积层。这里并没有执行什么计算，只是在TensorFlow图里添加了数学公式。
假设输入的是四维的张量，各个维度如下：

图像数量
每张图像的Y轴
每张图像的X轴
每张图像的通道数

输入通道可能是彩色通道，当输入是前面的卷积层生成的时候，它也可能是滤波通道。
输出是另外一个4通道的张量，如下：

图像数量，与输入相同
每张图像的Y轴。如果用到了2x2的池化，是输入图像宽高的一半。
每张图像的X轴。同上。
卷积滤波生成的通道数。


'''

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights

'''
转换一个层的帮助函数

卷积层生成了4维的张量。我们会在卷积层之后添加一个全连接层，
因此我们需要将这个4维的张量转换成可被全连接层使用的2维张量。
'''

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

'''
创建一个全连接层的帮助函数
这个函数为TensorFlow在计算图中创建了一个全连接层。这里也不进行任何计算，
只是往TensorFlow图中添加数学公式。
输入是大小为[num_images, num_inputs]的二维张量。输出是大小为[num_images,
num_outputs]的2维张量。


'''

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer
'''
占位符 （Placeholder）变量
Placeholder是作为图的输入，每次我们运行图的时候都可能会改变它们。将这个过程称为feeding placeholder变量，后面将会描述它。
首先我们为输入图像定义placeholder变量。这让我们可以改变输入到TensorFlow图中的图像。这也是一个张量（tensor），代表一个多维向量或矩阵。数据类型设置为float32，形状设为[None, img_size_flat]，None代表tensor可能保存着任意数量的图像，每张图象是一个长度为img_size_flat的向量



'''

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

'''
卷积层希望x被编码为4维张量，因此我们需要将它的形状转换至[num_images, img_height, img_width, num_channels]。注意img_height == img_width == img_size，如果第一维的大小设为-1， num_images的大小也会被自动推导出来。转换运算如下：

'''

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

'''
接下来我们为输入变量x中的图像所对应的真实标签定义placeholder变量。变量的形状是[None, num_classes]，这代表着它保存了任意数量的标签，每个标签是长度为num_classes的向量，本例中长度为10。

'''

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

'''
我们也可以为class-number提供一个placeholder，但这里用argmax来计算它。这里只是TensorFlow中的一些操作，没有执行什么运算。

'''
y_true_cls = tf.argmax(y_true, axis=1)

'''
卷积层 1

创建第一个卷积层。将x_image当作输入，创建num_filters1个不同的滤波器，
每个滤波器的宽高都与 filter_size1相等。最终我们会用2x2的max-pooling将图像降采样，使它的尺寸减半。

'''

layer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)

'''
检查卷积层输出张量的大小。它是（？,14, 14, 16），这代表着有任意数量的图像（？代表数量），
每张图像有14个像素的宽和高，有16个不同的通道，每个滤波器各有一个通道。

'''

layer_conv1

'''
卷积层 2

创建第二个卷积层，它将第一个卷积层的输出作为输入。输入通道的数量对应着第一个卷积层的滤波数。

'''

layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)

'''
核对一下这个卷积层输出张量的大小。它的大小是（？， 7， 7， 36）,其中？也代表着任意数量的图像，
每张图有7像素的宽高，每个滤波器有36个通道。
'''

layer_conv2

'''
转换层

这个卷积层输出一个4维张量。现在我们想将它作为一个全连接网络的输入，这就需要将它转换成2维张量。
    
'''

layer_flat, num_features = flatten_layer(layer_conv2)

'''
这个张量的大小是（？， 1764），意味着共有一定数量的图像，每张图像被转换成长为1764的向量。
其中1764 = 7 x 7 x 36
    
'''


layer_flat

num_features

'''
全连接层 1

往网络中添加一个全连接层。输入是一个前面卷积得到的被转换过的层。
全连接层中的神经元或节点数为fc_size。我们可以用ReLU来学习非线性关系。
'''

layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)


'''
全连接层的输出是一个大小为（？，128）的张量，？代表着一定数量的图像，并且fc_size == 128。
'''

layer_fc1

'''
全连接层 2

添加另外一个全连接层，它的输出是一个长度为10的向量，它确定了输入图是属于哪个类别。这层并没有用到ReLU。

'''

layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)

layer_fc2

'''
预测类别
第二个全连接层估算了输入图有多大的可能属于10个类别中的其中一个。然而，这是很粗略的估计并且很难解释，因为数值可能很小或很大，因此我们会对它们做归一化，将每个元素限制在0到1之间，并且相加为1。这用一个称为softmax的函数来计算的，结果保存在y_pred中。


'''

y_pred = tf.nn.softmax(layer_fc2)

#类别数字是最大元素的索引。

y_pred_cls = tf.argmax(y_pred, axis=1)

'''
优化损失函数
为了使模型更好地对输入图像进行分类，我们必须改变weights和biases变量。首先我们需要对比模型y_pred的预测输出和期望输出的y_true，来了解目前模型的性能如何。
交叉熵（cross-entropy）是在分类中使用的性能度量。交叉熵是一个常为正值的连续函数，如果模型的预测值精准地符合期望的输出，它就等于零。因此，优化的目的就是通过改变网络层的变量来最小化交叉熵。
TensorFlow有一个内置的计算交叉熵的函数。这个函数内部计算了softmax，所以我们要用layer_fc2的输出而非直接用y_pred,因为y_pred上已经计算了softmax。



'''
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)

'''

我们为每个图像分类计算了交叉熵，所以有一个当前模型在每张图上表现的度量。但是为了用交叉熵来指导模型变量的优化，我们需要一个额外的标量值，因此简单地利用所有图像分类交叉熵的均值。


'''

cost = tf.reduce_mean(cross_entropy)

'''
优化方法
既然我们有一个需要被最小化的损失度量，接着就可以建立优化一个优化器。这个例子中，我们使用的是梯度下降的变体AdamOptimizer。
优化过程并不是在这里执行。实际上，还没计算任何东西，我们只是往TensorFlow图中添加了优化器，以便之后的操作。


'''

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

'''
性能度量

我们需要另外一些性能度量，来向用户展示这个过程。

这是一个布尔值向量，代表预测类型是否等于每张图片的真实类型。

'''

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

#上面的计算先将布尔值向量类型转换成浮点型向量，这样子False就变成0，True变成1，然后计算这些值的平均数，以此来计算分类的准确度。

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

'''
运行TensorFlow
创建TensorFlow会话（session）

一旦创建了TensorFlow图，我们需要创建一个TensorFlow会话，用来运行图。
'''


session = tf.Session()

'''
初始化变量

我们需要在开始优化weights和biases变量之前对它们进行初始化。
'''

session.run(tf.global_variables_initializer())

'''
用来优化迭代的帮助函数
在训练集中有50,000张图。用这些图像计算模型的梯度会花很多时间。因此我们利用随机梯度下降的方法，它在优化器的每次迭代里只用到了一小部分的图像。
如果内存耗尽导致电脑死机或变得很慢，你应该试着减少这些数量，但同时可能还需要更优化的迭代。


'''

train_batch_size = 64

'''
函数执行了多次的优化迭代来逐步地提升网络层的变量。在每次迭代中，从训练集中选择一批新的数据，
然后TensorFlow用这些训练样本来执行优化器。每100次迭代会打印出相关信息。
'''

# Counter for total number of iterations performed so far.
total_iterations = 0

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

'''
用来绘制错误样本的帮助函数

函数用来绘制测试集中被误分类的样本。
'''

def plot_example_errors(cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.test.cls[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])

'''
绘制混淆（confusion）矩阵的帮助函数
'''

def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = data.test.cls
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

'''
展示性能的帮助函数
函数用来打印测试集上的分类准确度。
为测试集上的所有图片计算分类会花费一段时间，因此我们直接用这个函数来调用上面的结果，这样就不用每次都重新计算了。
这个函数可能会占用很多电脑内存，这也是为什么将测试集分成更小的几个部分。如果你的电脑内存比较小或死机了，就要试着降低batch-size。

'''

# Split the test-set into smaller batches of this size.
test_batch_size = 256

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = len(data.test.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.test.cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)


'''
优化之前的性能
测试集上的准确度很低，这是由于模型只做了初始化，并没做任何优化，所以它只是对图像做随机分类。
'''


print_test_accuracy()

'''
1次迭代后的性能
做了一次优化后，此时优化器的学习率很低，性能其实并没有多大提升。
'''


optimize(num_iterations=1)

print_test_accuracy()

'''

100次迭代优化后的性能
100次优化迭代之后，模型显著地提升了分类的准确度。
'''

optimize(num_iterations=99) # We already performed 1 iteration above.

print_test_accuracy(show_example_errors=True)

'''
1000次优化迭代后的性能
1000次优化迭代之后，模型在测试集上的准确度超过了90%。
'''
optimize(num_iterations=900) # We performed 100 iterations above.

print_test_accuracy(show_example_errors=True)

'''
10,000次优化迭代后的性能
经过10,000次优化迭代后，测试集上的分类准确率高达99%。
'''

optimize(num_iterations=9000) # We performed 1000 iterations above.

print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)

'''
权重和层的可视化
为了理解为什么卷积神经网络可以识别手写数字，我们将会对卷积滤波和输出图像进行可视化。

绘制卷积权重的帮助函数
'''

def plot_conv_weights(weights, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.
    
    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

'''
绘制卷积层输出的帮助函数
'''

def plot_conv_layer(layer, image):
    # Assume layer is a TensorFlow op that outputs a 4-dim tensor
    # which is the output of a convolutional layer,
    # e.g. layer_conv1 or layer_conv2.

    # Create a feed-dict containing just one image.
    # Note that we don't need to feed y_true because it is
    # not used in this calculation.
    feed_dict = {x: [image]}

    # Calculate and retrieve the output values of the layer
    # when inputting that image.
    values = session.run(layer, feed_dict=feed_dict)

    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i<num_filters:
            # Get the output image of using the i'th filter.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
'''
输入图像

绘制图像的帮助函数
'''

def plot_image(image):
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='binary')

    plt.show()

#如下所示，绘制一张测试集中的图像。

image1 = data.test.images[0]
plot_image(image1)

#绘制测试集里的另一张图像。
image2 = data.test.images[13]
plot_image(image2)


'''
卷积层 1

现在绘制第一个卷积层的滤波权重。

其中正值权重是红色的，负值为蓝色。
'''

plot_conv_weights(weights=weights_conv1)

'''
将这些卷积滤波添加到第一张输入图像，得到以下输出，它们也作为第二个卷积层的输入。
注意这些图像被降采样到14 x 14像素，即原始输入图分辨率的一半。
'''

plot_conv_layer(layer=layer_conv1, image=image1)

'''
下面是将卷积滤波添加到第二张图像的结果。
'''

plot_conv_layer(layer=layer_conv1, image=image2)

'''
从这些图像很难看出卷积滤波的作用是什么。显然，它们生成了输入图像的一些变体，就像光线从不同角度打到图像上并产生阴影一样。
卷积层 2
现在绘制第二个卷积层的滤波权重。
第一个卷积层有16个输出通道，代表着第二个卷基层有16个输入。第二个卷积层的每个输入通道也有一些权重滤波。我们先绘制第一个通道的权重滤波。
同样的，正值是红色，负值是蓝色。

'''


plot_conv_weights(weights=weights_conv2, input_channel=0)

'''
第二个卷积层共有16个输入通道，我们可以同样地画出其他图像。这里我们画出第二个通道的图像。
'''
plot_conv_weights(weights=weights_conv2, input_channel=1)

'''
由于这些滤波是高维度的，很难理解它们是如何应用的。

给第一个卷积层的输出加上这些滤波，得到下面的图像。

这些图像被降采样至7 x 7的像素，即上一个卷积层输出的一半。
'''

plot_conv_layer(layer=layer_conv2, image=image1)


#这是给第二张图像加上滤波权重的结果。

plot_conv_layer(layer=layer_conv2, image=image2)

'''
从这些图像来看，似乎第二个卷积层会检测输入图像中的线段和模式，这对输入图中的局部变化不那么敏感。

关闭TensorFlow会话

现在我们已经用TensorFlow完成了任务，关闭session，释放资源。
'''
# This has been commented out in case you want to modify and experiment
# with the Notebook without having to restart it.
# session.close()
