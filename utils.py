import numpy as np
import tensorflow as tf
from external.deuces.card import Card
from external.deuces.evaluator import Evaluator

card_type = { 'D': 0, 'H': 1, 'C': 2, 'S': 3 }
card_num = { 'A':1, 'T': 10, 'J': 11, 'Q': 12, 'K': 13 }
INITIAL_STACK = 200
action_num = {'FOLD': 0, 'SMALLBLIND': 1, 'BIGBLIND':2, 'CALL': 3, 'RAISE': 4}
street = { 'preflop': 0, 'flop': 1, 'river': 2, 'turn': 3 }
INITIAL_STACK = 200

def normalize_data(data, mi, ma):
    # (X - min) / (max - min)
    return (data - mi) / (ma - mi)

def encode_card(card):
    # Parse card
    try:
        num = int(card[1:])
    except Exception as e:
        num = card_num[card[1]]
    # 53 cards bc 0 means no card
    return normalize_data(card_type[card[0]] * 13 + num, 0, 52)

def encode_action(action, encoded_street, encoded_pot):
    """
    encoded_action:
     [0]: Street where this action was made.
     [1]: Action done by the player
     [2]: Total amount wagered by the player
     [3]: Amount wagered by the player on this action
     [4]: Increment on raise
     [5]: Pot before this action
    """
    # Create the placeholder for this action
    encoded_action = np.zeros(6)
    # Encode the street
    encoded_action[0] = encoded_street
    # Encode the action done
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
    # Encoded pot
    encoded_action[5] = encoded_pot
    return encoded_action

def encode_history(history):
    encoded = []
    pot = np.zeros(1)
    # Go through the actions of each street
    for key in dict.keys(history):
        # Encode the street
        encoded_street = normalize_data(street[key], 0, 3)
        for action in history[key]:
            # Encode this action
            encoded_action = encode_action(action, encoded_street, pot[0])
            # Update the pot value
            pot[0] += encoded_action[3]
            # Append the encoded action to the history
            encoded.append(encoded_action)
    # Return the encoded history
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
    # Instantiate Deuces Evaluator
    evaluator = Evaluator()
    # Converting PyPoker cards to Deuces card
    hand = [Card.new(card[1:] + card[0].lower()) for card in raw['hc']]
    board = [Card.new(card[1:] + card[0].lower()) for card in raw['cc']]
    # Hand strength calculated by Deuces
    hand_strength = evaluator.evaluate(hand, board) if len(board) > 0 else 7463

    # Encode hand cards
    data = [encode_card(card) for card in raw['hc']]
    # Encode position
    data.append(normalize_data(raw['pos'], 0, 5))
    # Encode Board cards
    [data.append(encode_card(card)) for card in raw['cc']]
    # Append 0 to represent unknow cards
    [data.append(0) for i in range(5 - len(raw['cc']))]
    # Encoding hand strength
    data.append(normalize_data(hand_strength, 1, 7463))
    # Encode button position
    data.append(normalize_data(raw['btn'], 0, 5))
    # Encode pot based on our Initial Stack
    data.append(normalize_data(raw['pot'], 0, 200))
    # Encode players data
    data.extend(encode_players(raw['sts'], raw['pos']))
    # Convert this to a tensor
    data = tf.convert_to_tensor([data], dtype=tf.float32)
    # Encode and convert to tensor the actions made before this action.
    history = tf.convert_to_tensor([encode_history(raw['history'])], dtype=tf.float32)

    return data, history
