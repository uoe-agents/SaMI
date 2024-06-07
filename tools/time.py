import signal
import wandb

def timeout_handler(signum, frame):
    # 自定义的超时处理函数
    raise TimeoutError("Execution timed out")

def check_timeout(func, kwargs, time = 10):
    try:
        # 在函数开始处设置希望的超时时间
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(1000)  # 设置超时时间为10秒

        func(**kwargs)

        # 如果在超时时间内完成了操作，则取消超时
        signal.alarm(0)
    except TimeoutError:
        print('wandb超时:', str(kwargs), '显示失败')
        pass

