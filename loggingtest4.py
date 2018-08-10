import logging

# 创建logger，如果参数为空则返回root logger
logger = logging.getLogger("nick")
logger.setLevel(logging.DEBUG)  # 设置logger日志等级

# 创建handler
fh = logging.FileHandler("test.log", encoding="utf-8")
ch = logging.StreamHandler()

# 设置输出日志格式
formatter = logging.Formatter(
    fmt="%(asctime)s %(name)s %(filename)s %(message)s",
    datefmt="%Y/%m/%d %X"
)

# 注意 logging.Formatter的大小写

# 为handler指定输出格式，注意大小写
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# 为logger添加的日志处理器
logger.addHandler(fh)
logger.addHandler(ch)

# 输出不同级别的log
logger.warning("泰拳警告")
logger.info("提示")
logger.error("错误")

