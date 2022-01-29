# @author: Ariel
# @time: 2021/4/3 14:40

class Foobar:
    def __init__(self):
        pass
    def __call__(self, *args, **kwargs):
        print('hello' + str(args[0]))

# 定义类对象
foobar = Foobar()
# 实例化类对象 因为该类实现了__call__()函数 所以实例化时会调用该函数
foobar(1,2,3)

# def func(*args, x, y):
#     print(args)

def func(*args, **kwargs):
    print(args)
    print(kwargs)


# 不确定个数的参数可以使用 - *args - 元组
# 指定参数名称的不确定个数的参数可以使用 - **kwargs - 字典
func(1,2,3,4,x=5,y=6)

