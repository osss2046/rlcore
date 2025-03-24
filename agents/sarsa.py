# rlcore/agents/sarsa.py
import numpy as np
import pickle
from core.agent import Agent
from agents.q_learning import EpsilonGreedyPolicy

class SARSAAgent(Agent):
    """
    Agente que implementa el algoritmo SARSA (State-Action-Reward-State-Action).
    A diferencia de Q-Learning, SARSA es un algoritmo on-policy que actualiza
    los valores Q basándose en la acción que realmente se tomará en el siguiente
    estado, no en la acción óptima.
    """
    
    def __init__(self, action_space, observation_space, learning_rate=0.1, 
                 discount_factor=0.99, epsilon=0.1, epsilon_decay=0.999, epsilon_min=0.01):
        """
        Inicializa el agente SARSA.
        
        Args:
            action_space: Espacio de acciones del entorno
            observation_space: Espacio de observaciones del entorno
            learning_rate: Tasa de aprendizaje (alpha)
            discount_factor: Factor de descuento (gamma)
            epsilon: Probabilidad inicial de exploración
            epsilon_decay: Factor de decaimiento de epsilon
            epsilon_min: Valor mínimo de epsilon
        """
        super().__init__(action_space, observation_space)
        
        # Determinar si estamos tratando con espacios discretos
        self._check_spaces()
        
        # Hiperparámetros
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Inicializar tabla Q con valores optimistas para fomentar exploración
        self.q_table = {}
        
        # Inicializar política epsilon-greedy
        self.policy = EpsilonGreedyPolicy(
            action_space, 
            self.q_table, 
            epsilon, 
            epsilon_decay, 
            epsilon_min
        )
        
        # Estado y acción actuales (necesarios para SARSA)
        self.current_state = None
        self.current_action = None
    
    def _check_spaces(self):
        """
        Verifica que los espacios de observación y acción sean compatibles con SARSA.
        """
        # Para action_space
        if hasattr(self.action_space, 'n'):
            self.n_actions = self.action_space.n
        elif isinstance(self.action_space, int):
            self.n_actions = self.action_space
        else:
            raise ValueError("SARSAAgent requiere un espacio de acciones discreto")
        
        # Para observation_space, aseguramos que sea discretizable
        if not hasattr(self.observation_space, 'n') and not isinstance(self.observation_space, int):
            # En este punto asumimos que el usuario proporcionará estados que pueden ser usados como claves
            pass
    
    def select_action(self, observation):
        """
        Selecciona una acción basada en la observación actual.
        
        Args:
            observation: La observación actual del entorno
            
        Returns:
            action: La acción seleccionada
        """
        # Al inicio de un episodio, guardamos el estado actual
        if self.current_state is None:
            self.current_state = observation
            self.current_action = self.policy.select_action(observation)
        
        return self.current_action
    
    def update(self, experience):
        """
        Actualiza la tabla Q con una nueva experiencia.
        
        Args:
            experience: Objeto Experience o tupla (state, action, reward, next_state, done)
        """
        # Extraer componentes de la experiencia
        if hasattr(experience, 'state'):
            state = experience.state
            action = experience.action
            reward = experience.reward
            next_state = experience.next_state
            done = experience.done
        else:
            state, action, reward, next_state, done = experience
        
        # Verificar consistencia con estado y acción actuales
        assert state == self.current_state
        assert action == self.current_action
        
        # Inicializar entradas en la tabla Q si no existen
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0.0
        
        # Seleccionar siguiente acción según la política actual (on-policy)
        if not done:
            next_action = self.policy.select_action(next_state)
            
            # Asegurar que la entrada Q existe para el siguiente par estado-acción
            if (next_state, next_action) not in self.q_table:
                self.q_table[(next_state, next_action)] = 0.0
                
            # Actualizar Q usando la ecuación de Bellman para SARSA
            # Q(s,a) := Q(s,a) + α * [r + γ * Q(s',a') - Q(s,a)]
            current_q = self.q_table[(state, action)]
            next_q = self.q_table[(next_state, next_action)]
            td_target = reward + self.discount_factor * next_q
            td_error = td_target - current_q
            self.q_table[(state, action)] += self.learning_rate * td_error
            
            # Actualizar estado y acción actuales para el próximo paso
            self.current_state = next_state
            self.current_action = next_action
        else:
            # Si es el final del episodio
            current_q = self.q_table[(state, action)]
            td_target = reward  # No hay estado siguiente
            td_error = td_target - current_q
            self.q_table[(state, action)] += self.learning_rate * td_error
            
            # Reiniciar estado y acción actuales
            self.current_state = None
            self.current_action = None
        
        # Actualizar epsilon para la política
        self.policy.update()
    
    def reset(self):
        """
        Reinicia el estado interno del agente al comienzo de un episodio.
        """
        # Reiniciar estado y acción actuales
        self.current_state = None
        self.current_action = None
    
    def save(self, filepath):
        """
        Guarda la tabla Q en un archivo.
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'params': {
                    'learning_rate': self.learning_rate,
                    'discount_factor': self.discount_factor,
                    'epsilon': self.policy.epsilon,
                    'epsilon_decay': self.policy.epsilon_decay,
                    'epsilon_min': self.policy.epsilon_min
                }
            }, f)
    
    def load(self, filepath):
        """
        Carga una tabla Q previamente guardada.
        
        Args:
            filepath: Ruta del modelo a cargar
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.learning_rate = data['params']['learning_rate']
            self.discount_factor = data['params']['discount_factor']
            self.policy.epsilon = data['params']['epsilon']
            self.policy.epsilon_decay = data['params']['epsilon_decay']
            self.policy.epsilon_min = data['params']['epsilon_min']
            
            # Reiniciar estado y acción actuales
            self.current_state = None
            self.current_action = None