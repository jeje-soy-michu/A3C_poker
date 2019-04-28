from worker import Worker
import multiprocessing
class Master:
    def train(self):
        workers = [Worker() for _ in range(multiprocessing.cpu_count())]
        for i, worker in enumerate(workers):
            print("Starting worker {}".format(i))
            worker.start()
    def play(self):
        print("Playing.")
