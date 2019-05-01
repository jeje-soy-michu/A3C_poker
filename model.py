import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.compat.v1.keras.layers import CuDNNLSTM

class Model(keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        self.lstm1 = CuDNNLSTM(128, return_sequences=True)
        self.lstm2 = CuDNNLSTM(64)
        self.history = Dense(32)


        self.dense1 = Dense(100, activation='relu')
        self.policy = Dense(2)

        self.dense2 = Dense(100, activation='relu')
        self.amount = Dense(1)

        self.dense3 = Dense(100, activation='relu')
        self.values = Dense(1)

    def call(self, inputs, history):
        h = self.lstm1(history)
        h = self.lstm2(h)
        h = self.history(h)

        inputs = tf.concat([inputs, h], 1)

        # Forward pass
        x = self.dense1(inputs)
        logits = self.policy(x)

        a = self.dense2(inputs)
        amount = self.amount(a)

        v = self.dense3(inputs)
        values = self.values(v)
        return logits, amount, values
