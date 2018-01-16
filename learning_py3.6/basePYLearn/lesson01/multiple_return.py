# coding:utf-8
'''
python 多返回值的使用方法 返回的是一个tuple，其中存放了各个返回值的顺序对象。
'''
def multiple_return_func():
    a = 'abc'
    b = 1
    c = 1.0
    return a, b, c

return_type = multiple_return_func()
print(type(return_type))
print(return_type[0], print(type(return_type[0])))
print(return_type[1], print(type(return_type[1])))
print(return_type[2], print(type(return_type[2])))