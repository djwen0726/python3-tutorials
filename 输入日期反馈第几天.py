import time
def inpu():
    the_year = int(input('输入年份(年份需大于等于1970小于等于2038):'))
    if len(str(the_year)) == 4 and 1970 <= the_year <= 2038:
        the_mon = int(input('输入月份:'))
        if 1 <= the_mon <= 12:
            the_day = int(input('输入日:'))
            if 1 <= the_mon <= 31:
                date_time = '%s-%s-%s' % (the_year, the_mon, the_day)
                print('你输入的日期为:', date_time)
    return date_time
def pan(dt):
    return time.strptime(dt, '%Y-%m-%d').tm_yday
a = inpu()
b = pan(a)
print('你输入的日期为当年的第%s天'%b)

'''
结果：
输入年份(年份需大于等于1970小于等于2038):2014
输入月份:1
输入日:31
你输入的日期为: 2014-1-31
你输入的日期为当年的第31天

'''
