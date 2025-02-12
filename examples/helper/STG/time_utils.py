import time
import functools
from contextlib import contextmanager

# 装饰器方式
def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f'{func.__name__} took {(end-start)*1000:.2f} ms')
        return result
    return wrapper

# 上下文管理器方式
@contextmanager
def timeblock(name):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f'{name} took {(end-start)*1000:.2f} ms')