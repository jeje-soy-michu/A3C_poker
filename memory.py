class Memory:
    def __init__(self):
        self.clear()

    def store(self, state, move, action, reward):
        self.states.append(state)
        self.moves.append(move)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.moves = []
        self.actions = []
        self.rewards = []
