import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.compat.v1.keras.layers import CuDNNLSTM

class Model(keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        # Model for History
        self.history = Sequential()
        self.history.add(CuDNNLSTM(128, return_sequences=True))
        self.history.add(CuDNNLSTM(128))
        self.history.add(Dense(64))

        # Model for Policy
        self.policy = Sequential()
        self.policy.add(Dense(128, activation='relu'))
        self.policy.add(Dense(128, activation='relu'))
        self.policy.add(Dense(64, activation='relu'))
        self.policy.add(Dense(2))

        # Model for Amount
        self.amount = Sequential()
        self.amount.add(Dense(128, activation='relu'))
        self.amount.add(Dense(128, activation='relu'))
        self.amount.add(Dense(64, activation='relu'))
        self.amount.add(Dense(1))

        # Model for Value
        self.values = Sequential()
        self.values.add(Dense(128, activation='relu'))
        self.values.add(Dense(1))

    def call(self, inputs, history):
        h = self.history(history)

        inputs = tf.concat([inputs, h], 1)

        # Forward pass
        logits = self.policy(inputs)
        amount = self.amount(inputs)
        values = self.values(inputs)

        return logits, amount, values
