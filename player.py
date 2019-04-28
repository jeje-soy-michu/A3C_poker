from numpy.random import randint

from pypokerengine.players import BasePokerPlayer

class A3CPlayer(BasePokerPlayer):
    def __init__(self, debug=False):
        self.debug = debug

    def _print(self, *msg):
        if self.debug:
            print(msg)

    def random_action(self, valid_actions):
        rand = randint(len(valid_actions))
        action = valid_actions[rand]
        if(action['action'] == "raise"):
            amount = action['amount']
            if amount['max'] == -1:
                return "call", valid_actions[1]['amount']
            else:
                min_amount, max_amount = action['amount']['min'], action['amount']['max']
                return "raise", randint(min_amount, max_amount)
        else:
            return action['action'], action['amount']

    def declare_action(self, valid_actions, hole_card, round_state):
        return self.random_action(valid_actions)

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
