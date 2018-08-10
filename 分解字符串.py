def f1(str):
     print(list(str))
     print(set(list(str)))
     print(list(set(list(str))))
     return (list(set(list(str))))


str = input('请输入字符串:')
print(f1(str))

