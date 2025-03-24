# examples/gridworld/__init__.py
"""
Ejemplos de uso de RLCore en entornos GridWorld.

Este módulo contiene ejemplos que demuestran el uso de varios
algoritmos de aprendizaje por refuerzo en entornos de cuadrícula:
- gridworld_q_learning.py: Implementación básica de Q-Learning
- compare_algorithms.py: Comparación de algoritmos (Q-Learning, SARSA, Monte Carlo, etc.)
"""

# Import main functions for easier access
from .gridworld_q_learning import main as run_q_learning_example
from .compare_algorithms import main as run_comparison_example