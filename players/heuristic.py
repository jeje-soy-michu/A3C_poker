from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate

class HeuristicPlayer(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"
    nb_player = 6
    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [raise_action_info, call_action_info, fold_action_info]
        call_action_info = valid_actions[1]
        fold_action_info = valid_actions[0]

        community_card = round_state['community_card']
        win_rate = estimate_hole_card_win_rate(nb_simulation=100, nb_player=self.nb_player,
                                               hole_card=gen_cards(hole_card),
                                               community_card=gen_cards(community_card))
        if win_rate > 1 / float(self.nb_player) + 0.1:
            action, amount = call_action_info["action"], call_action_info["amount"]
        else:
            action, amount = fold_action_info["action"], fold_action_info["amount"]
        return action, amount

    def receive_game_start_message(self, game_info):
        self.nb_player = game_info['player_num']

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass
