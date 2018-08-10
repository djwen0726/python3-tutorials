import json


data = {
    'name':'juhao',
    'age': 18,
    'feature': ['高', '富', '帅']
}

#indent=True,  sort_keys=True排序，   ensure_ascii=False #设置为非ascii解析
temp = json.dumps(data, indent=2, sort_keys=True,  ensure_ascii=False)	


print(temp)
result = json.loads(temp)		#把json数据转化为python的数据类型		
print(result)


#操作文件中的对象

file = open('1.json','w',encoding='utf-8')

json.dump(data,file)



file.close()

file = open('1.json','r',encoding='utf-8')

info = json.load(file)

print(info)

print(info['feature'][2])
