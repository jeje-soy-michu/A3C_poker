import numpy as np
import tensorflow as tf
from numpy.random import randint
from utils import format_data
from pypokerengine.players import BasePokerPlayer

INITIAL_STACK = 200

class A3CPlayer(BasePokerPlayer):
    def __init__(self, model, memory=None, name=None, debug=0):
        self.name = name
        if name is not None:
            self.memory = memory
        self.debug = debug
        self.model = model

    def _print(self, msg, verbosity=3):
        if self.debug >= verbosity:
            print(msg)

    def store_data(self, data, history, action, amount, reward):
        if self.name is not None:
            self.memory.store(data, history, action, amount, reward)

    def declare_action(self, valid_actions, hole_card, round_state):
        raw_data = {
            'hc': hole_card,
            'pos': round_state['next_player'],
            'st': round_state['street'],
            'cc': round_state['community_card'],
            'btn': round_state['dealer_btn'],
            #TODO: Don't ignore side pots.
            'pot': round_state['pot']['main']['amount'],
            'sts': round_state['seats'],
            'history': round_state['action_histories']
        }
        data, history = format_data(raw_data)
        logits, amount, _ = self.model(data, history)

        probs = tf.nn.softmax(logits)
        fold = np.random.choice(2, p=probs.numpy()[0])
        if not fold:
            amount = (amount.numpy() * INITIAL_STACK).round()[0, 0]
            self._print(f"Amount: {amount}")
            if amount < valid_actions[1]['amount'] or valid_actions[2]['amount']['max'] == -1:
                action, amount = valid_actions[1]['action'], valid_actions[1]['amount']
            elif valid_actions[2]['amount']['max'] < amount:
                action, amount = valid_actions[2]['action'], valid_actions[2]['amount']['max']
            else:
                action, amount = valid_actions[2]['action'], amount
        else:
            action, amount = valid_actions[0]['action'], 0

        # Calculate the reward
        reward = -amount / INITIAL_STACK
        # Store data before sending the action
        self.store_data(data, history, fold, amount/INITIAL_STACK, reward)
        self._print(f"Action: {action}, Amount: {amount}")

        # Send the action
        return action, amount


    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass
