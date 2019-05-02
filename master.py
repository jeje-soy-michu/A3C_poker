import os
import numpy as np
import pandas as pd
import multiprocessing
from model import Model
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from worker import Worker
from pypokerengine.api.game import setup_config, start_poker

from player import A3CPlayer
from players.call import CallPlayer
from players.fold import FoldPlayer
from players.random import RandomPlayer

class Master:
    def __init__(self, args):
        self.debug = args.debug
        self.gamma = args.gamma

        self._print(f"Debug level: {self.debug}.", 1)

        save_dir = args.save_dir
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            self._print(f"Creating directory: {save_dir}.")
            os.makedirs(save_dir)

        self._print(f"Creating main model.")
        self.global_model = Model()  # global network
        self._print(f"Creating main optimizer.")
        self.opt = tf.compat.v1.train.AdamOptimizer(args.lr, use_locking=True)

        self._print(f"Initializing model.")
        init_action = tf.convert_to_tensor(np.random.random((1, 17)), dtype=tf.float32)
        init_history = tf.convert_to_tensor(np.random.random((1, 2, 6)), dtype=tf.float32)
        self.global_model(init_action, init_history)

        model_path = os.path.join(save_dir, 'A3C_Model.h5')
        if not args.new_model and os.path.isfile(model_path):
            self._print(f"Loading model: {model_path}.")
            self.global_model.load_weights(model_path)

    def _print(self, msg, verbosity=3):
        if self.debug >= verbosity :
            print(f"[MASTER] {msg}")

    def train(self):
        workers = [Worker(i, self.global_model, self.opt, self.save_dir, gamma=self.gamma, debug=self.debug) for i in range(multiprocessing.cpu_count())]
        for i, worker in enumerate(workers):
            self._print("Starting worker {}".format(i), 1)
            worker.start()
        [w.join() for w in workers]
        self.play()

    def play(self):
        player = A3CPlayer(self.global_model)
        config = setup_config(max_round=1000, initial_stack=200, small_blind_amount=1)

        config.register_player(name="A3C", algorithm=player)
        config.register_player(name="CallPlayer1", algorithm=CallPlayer())
        config.register_player(name="CallPlayer2", algorithm=CallPlayer())
        config.register_player(name="CallPlayer3", algorithm=CallPlayer())
        config.register_player(name="RandomPlayer1", algorithm=RandomPlayer())
        config.register_player(name="RandomPlayer2", algorithm=RandomPlayer())

        d = None
        for i in range(25):
            self._print(f"Playing game: {i+1}", 3)
            game_result = start_poker(config, verbose=0)
            t = pd.DataFrame(game_result['players'])
            t['round'] = i
            if d is None:
                d = t
            else:
                d = pd.concat((d, t))

        print(d.groupby('name').mean()['stack'].sort_values())
