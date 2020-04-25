import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
    
    def update(self, old_params, gradient):
        if not hasattr(self, 'delta_params'):
            self.delta_params = np.zeros_like(old_params)
        
        self.delta_params = self.momentum*self.delta_params - self.learning_rate*gradient
        new_params = old_params + self.delta_params
        
        return new_params