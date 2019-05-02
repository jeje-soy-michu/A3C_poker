import os
import threading
import numpy as np
from time import time
import tensorflow as tf

from model import Model
from memory import Memory
from player import A3CPlayer
from pypokerengine.api.emulator import Emulator

PLAYERS = 6
STACK = 200
SMALL_BLIND = 1
ANTE = 0
GAMES = 25
MAX_ROUND = 1000

class Worker(threading.Thread):

    save_lock = threading.Lock()

    def __init__(self, id, global_model, opt, save_dir, gamma, debug=1):
        super(Worker, self).__init__()
        self.id = id
        self.emul = None
        self.memories = {}
        self.global_model = global_model
        self.opt = opt
        self.local_model = Model()
        self.debug = debug
        self.save_dir = save_dir
        self.gamma = gamma

    def _print(self, msg, verbosity=3):
        if self.debug >= verbosity :
            print(f"[WORKER:{self.id}] {msg}")

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
        self._print("Resetting environment.")
        [self.memories[key].clear() for key in self.memories]
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
        self._print("Finding round finished event.")
        event = events.pop()
        while event['type'] != 'event_round_finish':
            try:
                event = events.pop()
            except Exception as e:
                return None
        return event

    def run(self):
        # Metrics
        playing_times = []
        epoch_times = []

        for game in range(GAMES):
            if game % (GAMES//4) == 0:
                self._print(f"Playing Game: {game+1}", 1)
            else:
                self._print(f"Playing Game: {game+1}", 2)
            state = self.reset_env()
            total_loss = 0
            start_time = time()
            playing_time = 0

            for hand in range(MAX_ROUND):
                st_playing = time()
                state, events = self.emul.run_until_round_finish(state)
                playing_time += time() - st_playing
                self._print(f"Playing hand: {hand+1}")
                event = self.last_event(events)
                if not self.everyone_folds(event):
                    # Calculate gradient wrt to local model. We do so by tracking the
                    # variables involved in computing the loss by using tf.GradientTape
                    with tf.GradientTape() as tape:
                        total_loss = tf.add(total_loss, self.compute_loss(event))
                        if event is None or hand == MAX_ROUND-1:
                            self._print(f"Total Game Loss: {total_loss}.")
                            # Calculate local gradients
                            grads = tape.gradient(total_loss, self.local_model.trainable_weights)
                            # Push local gradients to global model
                            self.opt.apply_gradients(zip(grads, self.global_model.trainable_weights))
                            # Update local model with new weights
                            self.local_model.set_weights(self.global_model.get_weights())
                            break
                [self.memories[key].clear() for key in self.memories]
                if event is None:
                    break
                state = self.emul.start_new_round(state)[0]
            with Worker.save_lock:
                self._print(f"Saving model to {self.save_dir}", 3)
                self.global_model.save_weights(
                    os.path.join(self.save_dir, 'A3C_Model.h5'))
            # Metrics
            end_time = time() - start_time
            epoch_times.append(end_time)
            playing_times.append(playing_time)
            self._print("Epoch stats:", 2)
            self._print(f"Time playing: {playing_time:.2f}", 2)
            self._print(f"Total Time: {end_time:.2f}", 2)
        # Metrics
        self._print("TRAIN STATS:", 1)
        self._print(f"AVG time Playing: {np.mean(playing_times):.2f}", 1)
        self._print(f"AVG time per epoch: {np.mean(epoch_times):.2f}", 1)
        self._print("Training finished.", 1)

    def compute_loss(self, event):
        winners = event['winners']
        pot = event['round_state']['pot']['main']['amount']/STACK/len(winners)
        for winner in winners:
            if len(self.memories[winner['uuid']].states) == 0:
                return 0
            total_reward = pot
            # Get discounted rewards
            discounted_rewards = []
            for reward in self.memories[winner['uuid']].rewards[::-1]:
                total_reward = reward + total_reward * self.gamma
                discounted_rewards.append(total_reward)
            discounted_rewards.reverse()

            states = tf.convert_to_tensor(np.vstack(self.memories[winner['uuid']].states), dtype=tf.float32)

            last_move = self.memories[winner['uuid']].moves[-1]
            moves = np.zeros((len(self.memories[winner['uuid']].moves), last_move.shape[1], last_move.shape[2]))

            for i, move in enumerate(self.memories[winner['uuid']].moves):
                moves[i, -move.shape[1]:] = move.numpy()[0]

            moves = tf.convert_to_tensor(value=moves, dtype=tf.float32)
            logits, amounts, values = self.local_model(states, moves)

            #Get our advantages
            advantage = tf.convert_to_tensor(value=np.array(discounted_rewards)[:, None],dtype=tf.float32) - values

            # Value loss
            value_loss = advantage ** 2

            #Get our advantages
            amount_diff = tf.convert_to_tensor(value=np.array(self.memories[winner['uuid']].amounts)[:, None],dtype=tf.float32) - amounts

            # Value loss
            amount_loss = amount_diff ** 2

            policy = tf.nn.softmax(logits)
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(policy), logits=logits)

            act = tf.convert_to_tensor(value=np.array(self.memories[winner['uuid']].actions),dtype=tf.int32)

            policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=act, logits=logits)

            policy_loss *= tf.stop_gradient(advantage)
            policy_loss -= 0.01 * entropy

            total_loss = tf.reduce_mean(input_tensor=(0.5 * value_loss + amount_loss + policy_loss))
            return total_loss
