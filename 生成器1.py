def func1(num):
     i=1
     while i<=num:
          yield '第%s个hello'%i
          i += 1

a = func1(10)

for i in a:
     print(i)
     
