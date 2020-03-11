# Imports
import random
import math

def generate_random_configuration():
    params = {'learning_rate': random.uniform(0.01, 0.25),
              'subsample': random.uniform(0.5, 1.0),
              'reg_alpha': random.uniform(0.01, 0.5),
              'max_depth': math.floor(random.uniform(3, 15)),
              'gamma': math.floor(random.uniform(0, 10))
              }
    return params