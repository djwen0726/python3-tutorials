#内嵌画图
%matplotlib inline
import matplotlib # 注意这个也要import一次
import matplotlib.pyplot as plt
myfont = matplotlib.font_manager.FontProperties(fname=r'C:/Windows/Fonts/msyh.ttf') # 这一行
plt.plot((1,2,3),(4,3,-1))
plt.xlabel(u'横坐标',  fontproperties=myfont) # 这一段
plt.ylabel(u'纵坐标',  fontproperties=myfont) # 这一段
#plt.show() # 有了%matplotlib inline 就可以省掉plt.show()了
