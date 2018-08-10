class Person(object):
    def __init__(self, name, age, occupation):
        self.name =name
        self.age = age
        self.occupation = occupation

    

    def __call__(self, others):
        print( '我的 姓名 is %s' % self.name)
        print( '我的 年龄 is %s' % self.age)
        print( '我的 职业 is %s' % self.occupation)
        
        print( '对方 姓名 is %s' % others.name)
        print( '对方 年龄 is %s' % others.age)
        print( '对方 职业 is %s' % others.occupation)

    def __del__(self):
        print("__del__方法被调用")
        print("对象马上被干掉了...%s %s"% (self.name,self.age))
        

a1 = Person('Tom',20,'学生')
b1 = Person('Jerry',30,'教师')

a1(b1)

print('-------------------------------------')

b1(a1)

