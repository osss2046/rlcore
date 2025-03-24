# examples/bandit/__init__.py
"""
Ejemplos de uso de RLCore en entornos Multi-Armed Bandit.

Este módulo demuestra diferentes estrategias para resolver
el problema del bandido armado:
- epsilon-greedy con diferentes parámetros
- UCB (Upper Confidence Bound)
- Gradiente (con y sin línea base)
"""

# Import main function for easier access
from .bandit_example import main as run_bandit_example