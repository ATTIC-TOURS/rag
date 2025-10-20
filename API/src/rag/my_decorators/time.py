import time
import inspect
from colorama import Fore


def time_performance(name: str):
    def actual_decorator(func):
        if inspect.iscoroutinefunction(func):  
            async def async_wrapper(*args, **kwargs):
                start = time.time()
                result = await func(*args, **kwargs) 
                end = time.time()
                print(Fore.CYAN + f"🕰️ Elapsed time: {end - start:.2f} seconds for {name}")
                return result
            return async_wrapper
        else:  
            def sync_wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                end = time.time()
                print(Fore.CYAN + f"🕰️ Elapsed time: {end - start:.2f} seconds for {name}")
                return result
            return sync_wrapper
    return actual_decorator
