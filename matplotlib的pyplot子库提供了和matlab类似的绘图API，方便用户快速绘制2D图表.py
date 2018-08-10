# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 1000) #x是0-10的连续的1000个平均点
y = np.sin(x)
z = np.cos(x**2)

plt.figure(figsize=(8,4))  #matplotlib的pyplot子库提供了和matlab类似的绘图API，方便用户快速绘制2D图表

#下面的两行程序通过调用plot函数在当前的绘图对象中进行绘图：
plt.plot(x,y,label="$sin(x)$",color="red",linewidth=2)
plt.plot(x,z,"b--",label="$cos(x^2)$")
#x和y轴的标签
plt.xlabel("Time(s)")
plt.ylabel("Volt")
#图标的抬头
plt.title("PyPlot First Example")
#y轴的最大最小值
plt.ylim(-1.2,1.2)
#图标签
plt.legend()

plt.show()
'''
xlabel : 设置X轴的文字
ylabel : 设置Y轴的文字
title : 设置图表的标题
ylim : 设置Y轴的范围
legend : 显示图示
最后调用plt.show()显示出我们创建的所有绘图对象。

'''

