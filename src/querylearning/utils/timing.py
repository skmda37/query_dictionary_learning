import time


def time_spent(tic1, tag='', count=1):
    toc1 = time.process_time()
    print(f"time spend on {tag} method = {(toc1 - tic1)*100./count:.2f}ms")
    return