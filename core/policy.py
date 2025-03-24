# rlcore/core/policy.py
from abc import ABC, abstractmethod
import numpy as np

class Policy(ABC):
    """
    Clase base abstracta para políticas.
    Define cómo el agente selecciona acciones basadas en observaciones.
    """
    
    @abstractmethod
    def select_action(self, observation):
        """
        Selecciona una acción basada en la observación actual.
        
        Args:
            observation: La observación actual del entorno
            
        Returns:
            action: La acción seleccionada
        """
        pass
    
    def update(self, *args, **kwargs):
        """
        Actualiza la política con nueva información.
        """
        pass


class RandomPolicy(Policy):
    """
    Política que selecciona acciones aleatorias.
    Útil como línea base o para exploración inicial.
    """
    
    def __init__(self, action_space):
        """
        Inicializa la política aleatoria.
        
        Args:
            action_space: Espacio de acciones del entorno
        """
        self.action_space = action_space
    
    def select_action(self, observation):
        """
        Selecciona una acción aleatoria.
        
        Args:
            observation: La observación actual (ignorada)
            
        Returns:
            action: Una acción aleatoria dentro del espacio de acciones
        """
        if hasattr(self.action_space, 'sample'):
            return self.action_space.sample()
        elif isinstance(self.action_space, int):
            return np.random.randint(0, self.action_space)
        else:
            raise ValueError("Action space type not supported")