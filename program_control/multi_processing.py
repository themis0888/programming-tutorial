import multiprocessing 
import time, os 


def worker(procnum, return_dict):
    '''worker function'''
    cnt = 0
    while 1: 
        print('Worker {:02d} start Step {:03d}'.format(procnum, cnt))
        cnt += 1 
        time.sleep(1)
        return_dict[procnum] = cnt
        if cnt > 4 and procnum == 1: break


if __name__ == '__main__':
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(5):
        p = multiprocessing.Process(target=worker, args=(i, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    print(return_dict.values())