#!/usr/bin/python
# -*- coding: UTF-8 -*-

class FooParent(object):
    def __init__(self):
        self.parent = 'I\'m the parent.'
        print ('Parent',self.parent)

    def bar(self,message):
        print ("%s from Parent" % message)

class FooChild(FooParent):
    def __init__(self):
        # super(FooChild,self) 首先找到 FooChild 的父类（就是类 FooParent），然后把类B的对象 FooChild 转换为类 FooParent 的对象
        super(FooChild,self).__init__()
        print ('Child')

    def bar(self,message):
        super(FooChild, self).bar(message)
        print ('Child bar fuction')
        print (self.parent)

if __name__ == '__main__':
    fooParent = FooParent()
    fooChild = FooChild()
    fooChild.bar('HelloWorld')
    print(fooParent.__dict__)
    print(fooChild.__dict__)
    print(help(fooParent))
    print(help(fooChild))