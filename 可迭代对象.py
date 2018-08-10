from collections import Iterable, Iterator

str1 = 'adadad'
for i in str1:
     print(i)

str2 = iter(str1)

print(isinstance(str1,Iterable))
print(isinstance(str1,Iterator))

print(isinstance(str2,Iterable))

print(next(str2))

print(next(str2))

print(next(str2))

