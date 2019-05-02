class Memory:
    def __init__(self):
        self.clear()

    def store(self, state, move, action, amount, reward):
        self.states.append(state)
        self.moves.append(move)
        self.actions.append(action)
        self.amounts.append(amount)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.moves = []
        self.actions = []
        self.amounts = []
        self.rewards = []
