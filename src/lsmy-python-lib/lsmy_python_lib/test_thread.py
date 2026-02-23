import threading
import multiprocessing
import time
import sys

def count_heavy(n):
    while n > 0:
        n -= 1

COUNT = 50_000_000

# Single-thread
start = time.time()
count_heavy(COUNT)
count_heavy(COUNT)
print(f"Time single thread: {time.time() - start:.2f}s")

# Multi-thread
t1 = threading.Thread(target=count_heavy, args=(COUNT,))
t2 = threading.Thread(target=count_heavy, args=(COUNT,))

start = time.time()
t1.start()
t2.start()
t1.join()
t2.join()
print(f"Time multi thread: {time.time() - start:.2f}s")

# Check is disable GIL (Python 3.13+)
status = getattr(sys, '_is_gil_enabled', lambda: "Unknown")
print(f"Status GIL: {status()}")

if __name__ == "__main__":
    # Multi-process
    p1 = multiprocessing.Process(target=count_heavy, args=(COUNT,))
    p2 = multiprocessing.Process(target=count_heavy, args=(COUNT,))

    start = time.time()
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    print(f"Time multi-process: {time.time() - start:.2f}s")

    print(f"Number of cores: {multiprocessing.cpu_count()}")