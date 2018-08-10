#---coding:utf-8---
from tkinter import * #导包
def resize(ev=None):
    '根据进度条调整字体大小'
    label.config(font='Helvetica -%d bold' %scale.get())

def writefile():
    '写文件'
    try:
        f = open(r'd:\hello.txt','w')
        f.write('hello,world!')
    except Exception as e:
        print(e)
    finally:
        f.close()

top = Tk()#新建一个窗口
top.geometry('400x300')#指定窗口大小
top.title('GUI_test')

label = Label(top,text='Hello,World!',font='Helvetica -12 bold')#随进度条变化的标签，刚开始学当然用hello，world
label.pack(fill=Y,expand=1)

scale = Scale(top,from_=10,to=50,orient=HORIZONTAL,command=resize)#进度条，个人认为command作用和绑定差不多
scale.set(12)#设初值
scale.pack(fill=X,expand=1)

write = Button(top,text="Write",command=writefile)
write.pack()

quit = Button(top,text="Quit",command=top.quit,activeforeground='White',
              activebackground='red')
quit.pack()

mainloop()#调用该函数运行程序