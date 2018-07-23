import multiprocessing 

def worker(num):
    """thread worker function"""
    print('Worker: %d'% num)
    return

if __name__ == '__main__':
    jobs = []
    for i in range(20):
        p = multiprocessing.Process(target=worker, args=(i,))
        jobs.append(p)
        p.start()