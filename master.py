from worker import Worker
import multiprocessing
import tensorflow as tf
from model import Model
import numpy as np

LEARNING_RATE = 3e-3

class Master:
    def __init__(self):
        self.opt = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE, use_locking=True)
        self.global_model = Model()  # global network
        init_action = tf.convert_to_tensor(np.random.random((1, 16)), dtype=tf.float32)
        init_history = tf.convert_to_tensor(np.random.random((1, 2, 6)), dtype=tf.float32)
        self.global_model(init_action, init_history)
    def train(self):
        workers = [Worker(i, self.global_model, self.opt) for i in range(multiprocessing.cpu_count())]
        for i, worker in enumerate(workers):
            print("Starting worker {}".format(i))
            worker.start()

    def play(self):
        print("Playing.")
