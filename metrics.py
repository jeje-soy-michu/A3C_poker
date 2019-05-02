from time import time
class Metrics:
    def __init__(self):
        self.metrics = {}

    def get(self, name):
        return self.metrics[name]

    def register(self, name):
        self.metrics[name] = []

    def reset(self, name):
        self.register(name)

    def store(self, name, value):
        self.metrics[name].append(value)

    def start(self, name):
        self.metrics[name].append(-time())

    def stop(self, name):
        self.metrics[name][-1] += time()
