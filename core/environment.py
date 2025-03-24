from abc import ABC, abstractmethod
import numpy as np

class Environment(ABC):
    """
    Clase base abstracta para entornos de RL.
    Define la interfaz estándar que todos los entornos deben implementar.
    """
    
    @abstractmethod
    def reset(self):
        """
        Reinicia el entorno a un estado inicial y devuelve la observación inicial.
        
        Returns:
            observation: Representación inicial del estado del entorno
        """
        pass
    
    @abstractmethod
    def step(self, action):
        """
        Ejecuta una acción en el entorno y avanza un paso de tiempo.
        
        Args:
            action: La acción a ejecutar
            
        Returns:
            observation: La nueva observación del entorno
            reward: La recompensa obtenida
            done: Indicador booleano de si el episodio ha terminado
            info: Diccionario con información adicional para depuración
        """
        pass
    
    @abstractmethod
    def render(self, mode='human'):
        """
        Renderiza el entorno para visualización.
        
        Args:
            mode: Modo de renderizado ('human', 'rgb_array', etc.)
            
        Returns:
            Depende del modo: None para 'human', np.array para 'rgb_array'
        """
        pass
    
    @property
    @abstractmethod
    def action_space(self):
        """
        Define el espacio de acciones válidas.
        
        Returns:
            Un objeto Space que define las acciones válidas
        """
        pass
    
    @property
    @abstractmethod
    def observation_space(self):
        """
        Define el espacio de observaciones válidas.
        
        Returns:
            Un objeto Space que define las observaciones posibles
        """
        pass
    
    def close(self):
        """
        Limpia los recursos utilizados por el entorno.
        """
        pass
    
    def seed(self, seed=None):
        """
        Establece las semillas para los generadores de números aleatorios.
        
        Args:
            seed: La semilla a utilizar
            
        Returns:
            La lista de semillas utilizadas
        """
        return [seed]