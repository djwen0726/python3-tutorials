import logging

# 第一步，创建一个logger
logger = logging.getLogger('%s_log' % __name__)
logger.setLevel(logging.INFO)  #设定日志等级

# 第二步，创建一个handler，用于写入日志文件
fh = logging.FileHandler('test1.log', mode='a',encoding='utf8')
fh.setLevel(logging.WARNING)

#输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)

# 第三步，定义handler的输出格式
#(时间，文件名，出错行数， 等级， 内容)
formatter = logging.Formatter("%(asctime)s - %(filename)s [line: %(lineno)d] - %(levelname)s: %(message)s") #第二个参数时间格式话
fh.setFormatter(formatter)
ch.setFormatter(formatter)


#第四步，将对应的hangler添加在logger对象中
logger.addHandler(fh)
logger.addHandler(ch)



try:
    with open('test.txt', 'r') as f:
        print(f.read())
except FileNotFoundError as e:
    print(e)
    #logger.error("No such file or directory: 'test.txt'")
    logger.error('No such file or directory')
