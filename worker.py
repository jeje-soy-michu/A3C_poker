import threading

import numpy as np
import tensorflow as tf

from model import Model
from memory import Memory
from player import A3CPlayer
from pypokerengine.api.emulator import Emulator

PLAYERS = 6
MAX_ROUND = 1000
STACK = 200
SMALL_BLIND = 1
ANTE = 0
GAMMA = .99

class Worker(threading.Thread):
    def __init__(self, id, global_model, opt):
        super(Worker, self).__init__()
        self.id = id
        self.emul = None
        self.memories = {}
        self.global_model = global_model
        self.opt = opt
        self.local_model = Model()
        self.debug = True

    def _print(self, msg):
        if self.debug:
            print(f"[{self.id}] - {msg}")

    def setup_emul(self):
        self._print("Setting up Emulator.")
        self.emul = Emulator()
        self.emul.set_game_rule(PLAYERS, MAX_ROUND, SMALL_BLIND, ANTE)
        players_info = {}
        for i in range(PLAYERS):
            self.memories[f"{i}"] = Memory()
            player = A3CPlayer(self.local_model, self.memories[f"{i}"], name=f"{i}")
            self.emul.register_player(f"{i}", player)
            players_info[f"{i}"] = { "name": f"A3C-{i}", "stack": STACK }

        self.initial_state = self.emul.generate_initial_game_state(players_info)

    def reset_env(self):
        if self.emul is None:
            self.setup_emul()
        return self.emul.start_new_round(self.initial_state)[0]

    def everyone_folds(self, event):
        if event is None:
            return True
        aux = True
        useless = ['SMALLBLIND', 'BIGBLIND', 'FOLD']
        for action in event['round_state']['action_histories']['preflop']:
            aux = aux and action['action'] in useless
        return aux

    def last_event(self, events):
        event = events.pop()
        while event['type'] != 'event_round_finish':
            try:
                event = events.pop()
            except Exception as e:
                return None
        return event

    def run(self):
        state = self.reset_env()
        while True:
            state, events = self.emul.run_until_round_finish(state)
            event = self.last_event(events)
            if not self.everyone_folds(event):
                # Calculate gradient wrt to local model. We do so by tracking the
                # variables involved in computing the loss by using tf.GradientTape
                with tf.GradientTape() as tape:
                    total_loss = self.compute_loss(event)
                    # Calculate local gradients
                    grads = tape.gradient(total_loss, self.local_model.trainable_weights)
                    # Push local gradients to global model
                    self.opt.apply_gradients(zip(grads, self.global_model.trainable_weights))
                    # Update local model with new weights
                    self.local_model.set_weights(self.global_model.get_weights())
            [self.memories[key].clear() for key in self.memories]
            if event is None:
                break
            state = self.emul.start_new_round(state)[0]
        print("Finished")

    def compute_loss(self, event):
        winners = event['winners']
        pot = event['round_state']['pot']['main']['amount']/STACK/len(winners)
        for winner in winners:
            total_reward = pot
            # Get discounted rewards
            discounted_rewards = []
            for reward in self.memories[winner['uuid']].rewards[::-1]:
                total_reward = reward + total_reward*GAMMA
                discounted_rewards.append(total_reward)
            discounted_rewards.reverse()

            states = tf.convert_to_tensor(np.vstack(self.memories[winner['uuid']].states), dtype=tf.float32)

            last_move = self.memories[winner['uuid']].moves[-1]
            moves = np.zeros((len(self.memories[winner['uuid']].moves), last_move.shape[1], last_move.shape[2]))

            for i, move in enumerate(self.memories[winner['uuid']].moves):
                moves[i, -move.shape[1]:] = move.numpy()[0]

            moves = tf.convert_to_tensor(value=moves, dtype=tf.float32)
            logits, amount, values = self.local_model(states, moves)

            #Get our advantages
            advantage = tf.convert_to_tensor(value=np.array(discounted_rewards)[:, None],dtype=tf.float32) - values

            # Value loss
            value_loss = advantage ** 2

            policy = tf.nn.softmax(logits)
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(policy), logits=logits)

            act = tf.convert_to_tensor(value=np.array(self.memories[winner['uuid']].actions),dtype=tf.int32)

            policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=act, logits=logits)

            policy_loss *= tf.stop_gradient(advantage)
            policy_loss -= 0.01 * entropy
            total_loss = tf.reduce_mean(input_tensor=(0.5 * value_loss + policy_loss))
            return total_loss
