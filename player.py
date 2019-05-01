import numpy as np
import tensorflow as tf
from numpy.random import randint

from pypokerengine.players import BasePokerPlayer


street = { 'preflop': 0, 'flop': 1, 'river': 2, 'turn': 3 }
card_type = { 'D': 0, 'H': 1, 'C': 2, 'S': 3 }
card_num = { 'A':1, 'T': 10, 'J': 11, 'Q': 12, 'K': 13 }
action_num = {'FOLD': 0, 'SMALLBLIND': 1, 'BIGBLIND':2, 'CALL': 3, 'RAISE': 4}

# BigBlind * 100 = 2 * 100 = 200
INITIAL_STACK = 200

def normalize_data(data, mi, ma):
    return (data - mi) / (ma - mi)

def encode_card(card):
    try:
        num = int(card[1:])
    except Exception as e:
        num = card_num[card[1]]
    # 53 cards bc 0 means no card
    return normalize_data(card_type[card[0]] * 13 + num, 0, 52)

def encode_action(action, encoded_street, pot):
    """
    encoded_action:
     [0]: Street where this action was made.
     [1]: Action done by the player
     [2]: Total amount wagered by the player
     [3]: Amount wagered by the player on this action
     [4]: Increment on raise
     [5]: Pot before this action
    """
    encoded_action = np.zeros(6)
    encoded_action[0] = encoded_street
    encoded_action[1] = normalize_data(action_num[action['action']], 0, 4)

    act = action['action']
    if act == 'RAISE':
        encoded_action[2] = normalize_data(action['amount'], 0, INITIAL_STACK)
        encoded_action[3] = normalize_data(action['paid'], 0, INITIAL_STACK)
        encoded_action[4] = normalize_data(action['add_amount'], 0, INITIAL_STACK)
    elif act == 'CALL':
        encoded_action[2] = normalize_data(action['amount'], 0, INITIAL_STACK)
        encoded_action[3] = normalize_data(action['paid'], 0, INITIAL_STACK)
    elif act == 'SMALLBLIND':
        encoded_action[2] = normalize_data(action['amount'], 0, INITIAL_STACK)
        encoded_action[3] = normalize_data(action['add_amount'], 0, INITIAL_STACK)
    elif act == 'BIGBLIND':
        encoded_action[2] = normalize_data(action['amount'], 0, INITIAL_STACK)
        encoded_action[3] = normalize_data(action['add_amount'] + 1, 0, INITIAL_STACK)
    encoded_action[5] = pot
    return encoded_action

def encode_history(history):
    encoded = []
    pot = np.zeros(1)
    for key in dict.keys(history):
        encoded_street = normalize_data(street[key], 0, 3)
        for action in history[key]:
            encoded_action = encode_action(action, encoded_street, pot[0])
            pot[0] += encoded_action[3]
            encoded.append(encoded_action)
    return encoded

def encode_players(players, pos):
    encoded = []
    for player in players:
        # Check if they are participating
        if player['state'] == "participating":
            # Normalizing their stack
            encoded += [normalize_data(player['stack'], 0, INITIAL_STACK)]
        else:
            # If they are not participating insert null values
            encoded += [0]
    # Return values ordered by our position
    return encoded[pos:] + encoded[:pos]

def format_data(raw):
    data = [encode_card(card) for card in raw['hc']]
    data.append(normalize_data(raw['pos'], 0, 5))
    [data.append(encode_card(card)) for card in raw['cc']]
    [data.append(0) for i in range(5 - len(raw['cc']))]
    data.append(normalize_data(raw['btn'], 0, 5))
    data.append(normalize_data(raw['pot'], 0, 200))
    data.extend(encode_players(raw['sts'], raw['pos']))
    data = tf.convert_to_tensor([data], dtype=tf.float32)
    history = tf.convert_to_tensor([encode_history(raw['history'])], dtype=tf.float32)
    return data, history

class A3CPlayer(BasePokerPlayer):
    def __init__(self, model, memory=None, name=None, debug=False):
        if name is not None:
            self.name = name
            self.memory = memory
        self.debug = debug
        self.model = model

    def _print(self, msg):
        if self.debug:
            print(msg)

    def random_action(self, valid_actions):
        rand = randint(len(valid_actions))
        action = valid_actions[rand]
        if(action['action'] == "raise"):
            amount = action['amount']
            if amount['max'] == -1:
                return "call", valid_actions[1]['amount']
            elif amount['max'] == amount['min']:
                return "raise", amount['max']
            else:
                min_amount, max_amount = action['amount']['min'], action['amount']['max']
                return "raise", randint(min_amount, max_amount)
        else:
            return action['action'], action['amount']

    def store_data(self, data, history, action, reward):
        if self.name is not None:
            self.memory.store(data, history, action, reward)

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
            amount = amount.numpy().round()
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
        self.store_data(data, history, fold, reward)
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
