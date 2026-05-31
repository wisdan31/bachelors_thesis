class Agent:
    def __init__(self, policy):
        self.policy = policy

    def act(self, observation):
        return self.policy.select(observation)
    
    