# rlcore/core/agent.py
from abc import ABC, abstractmethod

class Agent(ABC):
    """
    Clase base abstracta para agentes de RL.
    Define la interfaz que todos los agentes deben implementar.
    """
    
    def __init__(self, action_space, observation_space):
        """
        Inicializa un agente.
        
        Args:
            action_space: Espacio de acciones del entorno
            observation_space: Espacio de observaciones del entorno
        """
        self.action_space = action_space
        self.observation_space = observation_space
    
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
    
    @abstractmethod
    def update(self, experience):
        """
        Actualiza el agente con una nueva experiencia.
        
        Args:
            experience: Objeto Experience o tupla (state, action, reward, next_state, done)
        """
        pass
    
    def reset(self):
        """
        Reinicia el estado interno del agente al comienzo de un episodio.
        """
        pass
    
    def save(self, filepath):
        """
        Guarda el modelo del agente.
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        raise NotImplementedError
    
    def load(self, filepath):
        """
        Carga un modelo previamente guardado.
        
        Args:
            filepath: Ruta del modelo a cargar
        """
        raise NotImplementedError