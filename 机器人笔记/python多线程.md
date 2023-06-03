# python 全局解释器锁（GIL）
Python的多线程，只有用于I/O密集型程序时效率才会有明显的提高。

原因如下：

Python代码的执行是由Python虚拟机进行控制。它在主循环中同时只能有一个控制线程在执行，意思就是Python解释器中可以运行多个线程，但是在执行的只有一个线程，其他的处于等待状态。

这些线程执行是有全局解释器锁（GIL）控制，它来保证同时只有一个线程在运行。在多线程运行环境中，Python虚拟机执行方式如下：

1. 设置GIL
2. 切换进线程
3. 执行下面操作之一
4. 运行指定数量的字节码指令
5. 线程主动让出控制权
6. 切换出线程（线程处于睡眠状态）
7. 解锁GIL
8. 进入1步骤
                
    Python运行计算密集型的多线程程序时，更倾向于让线程在整个时间片内始终占据GIL，而I/O秘籍型的多线程程序在I/O被调用前会释放GIL，以允许其他线程在I/O执行的时候运行
# threading 模块的Thread 类
Thread 实例化时需要接收 target，args（kwargs）两个参数;

- target 用于接收需要使用多线程调用的对象
- args 或 kwargs 用于接收调用对象的需要用到的参数，args接收tuple，kwargs接收dict
```
from threading import Thread
from time import sleep, ctime

def func(name, sec):
    print('---开始---', name, '时间', ctime())
    sleep(sec)
    print('***结束***', name, '时间', ctime())

# 创建 Thread 实例
t1 = Thread(target=func, args=('第一个线程', 1))
t2 = Thread(target=func, args=('第二个线程', 2))

# 启动线程运行
t1.start()
t2.start()

# 等待所有线程执行完毕
t1.join()  # join() 等待线程终止，要不然一直挂起
t2.join()
```
输出
```
---开始--- 第一个线程 时间 Mon May 29 20:43:08 2023
---开始--- 第二个线程 时间 Mon May 29 20:43:08 2023
***结束*** 第一个线程 时间 Mon May 29 20:43:09 2023
***结束*** 第二个线程 时间 Mon May 29 20:43:10 2023
```
start()方法用来启动线程的执行。

join() 方法是一种自旋锁，它用来等待线程终止。也可以提供超时的时间，当线程运行达到超时时间后结束线程，如join(500)，500毫秒后结束线程运行

    注意：如果当你的主线程还有其他事情要做，而不是等待这些线程完成，就可以不调用join()。
    join()方法只有在你需要等待线程完成然后在做其他事情的时候才是有用的
# 派生Thread 的子类，并创建子类的实例
```
from threading import Thread
from time import sleep, ctime


# 创建 Thread 的子类
class MyThread(Thread):
    def __init__(self, func, args):
        '''
        :param func: 可调用的对象
        :param args: 可调用对象的参数
        '''
        Thread.__init__(self)   # 不要忘记调用Thread的初始化方法
        self.func = func
        self.args = args

    def run(self):
        self.func(*self.args)


def func(name, sec):
    print('---开始---', name, '时间', ctime())
    sleep(sec)
    print('***结束***', name, '时间', ctime())

def main():
    # 创建 Thread 实例
    t1 = MyThread(func, (1, 1))
    t2 = MyThread(func, (2, 2))
    # 启动线程运行
    t1.start()
    t2.start()
    # 等待所有线程执行完毕
    t1.join()
    t2.join()

if __name__ == '__main__':
    main()
```
# 获取可调用对象的返回值
- 在多线程中运行的程序时与主线程分开，我们没法直接获取线程中程序的返回值。这时就可以使用派生Thread 的子类，将给过保存的实例属性中，通过一个新方法获取运行结果
```
from threading import Thread
from time import sleep, ctime

# 创建 Thread 的子类
class MyThread(Thread):
    def __init__(self, func, args):
        '''
        :param func: 可调用的对象
        :param args: 可调用对象的参数
        '''
        Thread.__init__(self)
        self.func = func
        self.args = args
        self.result = None

    def run(self):
        self.result = self.func(*self.args)

    def getResult(self):
        return self.result


def func(name, sec):
    print('---开始---', name, '时间', ctime())
    sleep(sec)
    print('***结束***', name, '时间', ctime())
    return sec


def main():
    # 创建 Thread 实例
    t1 = MyThread(func, (1, 1))
    t2 = MyThread(func, (2, 2))
    # 启动线程运行
    t1.start()
    t2.start()
    # 等待所有线程执行完毕
    t1.join()
    t2.join()
    # 或线程中程序的运行结果
    print(t1.getResult())
    print(t2.getResult())


if __name__ == '__main__':
    main()
```
# Lock 同步锁
lock = threading.Lock()

lock.acquire()

lock.release()

    1. 同时存在两个线程：线程A，线程B
    2. 线程A 抢占到GIL，进入CPU执行，并加了Lock，但为执行完毕，线程被释放
    3. 线程B 抢占到GIL，进入CPU执行，执行时发现数据被线程A Lock，于是线程B被阻塞
    4. 线程B的GIL被夺走，有可能线程A拿到GIL，执行完操作、解锁，并释放GIL
    5. 线程B再次拿到GIL，才可以正常执行
- 通过上述应该能看到，Lock 通过牺牲执行的效率换数据安全
# 重入锁(递归锁) threading.RLock()
- 为了支持同一个线程中多次请求同一资源，Python 提供了可重入锁(RLock)。这个RLock内部维护着一个锁(Lock)和一个计数器(counter)变量，counter 记录了acquire 的次数，从而使得资源可以被多次acquire。直到一个线程所有 acquire都被release(计数器counter变为0)，其他的线程才能获得资源。
# 信号量 threading.Semaphore(k)
- semaphore.acquire()
- semaphore.release()
# Condition 条件变量
# Event 事件锁对象
- 用于线程间通信，即程序中的其一个线程需要通过判断某个线程的状态来确定自己下一步的操作，就用到了event()对象。event()对象有个状态值，他的默认值为 Flase，即遇到 event() 对象就阻塞线程的执行。
```
wait(timeout=None) 挂起线程timeout秒(None时间无限)，直到超时或收到event()信号开关为True时才唤醒程序。
set() Even状态值设为True
clear() Even状态值设为 False
isSet() 返回Even对象的状态值。
```
```
import threading

event = threading.Event()


def func():
    print('等待服务响应...')
    event.wait()  # 等待事件发生
    print('连接到服务')


def connect():
    print('成功启动服务')
    event.set()


t1 = threading.Thread(target=func, args=())
t2 = threading.Thread(target=connect, args=())

t1.start()
t2.start()
```
    输出结果:

    等待服务响应...
    成功启动服务
    连接到服务器

观察结果会发现，t1线程运行的func函数需要等到connect函数运行event.set()后才继续执行之后的操作。
# Barrie 障碍锁