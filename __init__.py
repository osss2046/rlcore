# rlcore/__init__.py
__version__ = '0.1.0'

# rlcore/agents/__init__.py
from .agents.q_learning import QLearningAgent, EpsilonGreedyPolicy

# rlcore/training/__init__.py
from training.trainer import Trainer