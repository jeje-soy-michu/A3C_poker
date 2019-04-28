import threading

from player import A3CPlayer
from pypokerengine.api.game import setup_config, start_poker

PLAYERS = 6
MAX_ROUND = 1000
STACK = 200
SMALL_BLIND = 1

class Worker(threading.Thread):
    def __init__(self):
        super(Worker, self).__init__()

    def reset_env(self):
        config = setup_config(max_round=MAX_ROUND, initial_stack=STACK, small_blind_amount=SMALL_BLIND)
        [config.register_player(name=str(i+1), algorithm=A3CPlayer()) for i in range(PLAYERS)]
        return config

    def run(self):
        start_state = self.reset_env()
        start_poker(start_state, verbose=1)
