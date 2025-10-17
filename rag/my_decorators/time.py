import time


def time_performance(name: str):
    def actual_decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            elapsed = end - start
            print(f"🕰️ Elapsed time: {elapsed:.2f} seconds for {name}")
            return result
        return wrapper
    return actual_decorator