# rlcore/agents/q_learning.py
import numpy as np
from core.agent import Agent
from core.policy import Policy
import pickle

class EpsilonGreedyPolicy(Policy):
    """
    Política epsilon-greedy que balancea exploración y explotación.
    Con probabilidad epsilon selecciona una acción aleatoria,
    y con probabilidad 1-epsilon selecciona la acción con mayor valor Q.
    """
    
    def __init__(self, action_space, q_function, epsilon=0.1, epsilon_decay=1.0, epsilon_min=0.01):
        """
        Inicializa la política epsilon-greedy.
        
        Args:
            action_space: Espacio de acciones del entorno
            q_function: Función Q para valorar acciones (puede ser dict o array)
            epsilon: Probabilidad de exploración (0 <= epsilon <= 1)
            epsilon_decay: Factor de decaimiento de epsilon por cada actualización
            epsilon_min: Valor mínimo de epsilon
        """
        self.action_space = action_space
        self.q_function = q_function
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Determinar si action_space es discreto y obtener su tamaño
        if hasattr(action_space, 'n'):
            self.n_actions = action_space.n
        else:
            self.n_actions = action_space  # Asumimos que es un entero
    
    def select_action(self, state):
        """
        Selecciona una acción usando la política epsilon-greedy.
        
        Args:
            state: Estado actual del entorno
            
        Returns:
            action: La acción seleccionada
        """
        # Exploración: con probabilidad epsilon, seleccionar acción aleatoria
        if np.random.random() < self.epsilon:
            if hasattr(self.action_space, 'sample'):
                return self.action_space.sample()
            else:
                return np.random.randint(0, self.n_actions)
        
        # Explotación: seleccionar la acción con mayor valor Q
        if isinstance(self.q_function, dict):
            # Si Q es un diccionario, obtener valores para el estado actual
            q_values = [self.q_function.get((state, a), 0.0) for a in range(self.n_actions)]
        else:
            # Si Q es un array, obtener valores directamente
            q_values = self.q_function[state]
        
        # Manejar empates aleatoriamente
        max_q = np.max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return np.random.choice(best_actions)
    
    def update(self):
        """
        Actualiza epsilon según la tasa de decaimiento.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class QLearningAgent(Agent):
    """
    Agente que implementa el algoritmo Q-Learning.
    Usa una tabla Q para almacenar los valores estado-acción.
    """
    
    def __init__(self, action_space, observation_space, learning_rate=0.1, 
                 discount_factor=0.99, epsilon=0.1, epsilon_decay=0.999, epsilon_min=0.01):
        """
        Inicializa el agente Q-Learning.
        
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
    
    def _check_spaces(self):
        """
        Verifica que los espacios de observación y acción sean compatibles con Q-Learning.
        """
        # Para action_space
        if hasattr(self.action_space, 'n'):
            self.n_actions = self.action_space.n
        elif isinstance(self.action_space, int):
            self.n_actions = self.action_space
        else:
            raise ValueError("QLearningAgent requiere un espacio de acciones discreto")
        
        # Para observation_space, aseguramos que sea discretizable
        # (Q-Learning tabular trabaja con estados discretos o discretizables)
        if not hasattr(self.observation_space, 'n') and not isinstance(self.observation_space, int):
            # En este punto asumimos que el usuario proporcionará estados que pueden ser usados como claves
            # (por ejemplo, tuplas de enteros o strings)
            pass
    
    def select_action(self, observation):
        """
        Selecciona una acción basada en la observación actual.
        
        Args:
            observation: La observación actual del entorno
            
        Returns:
            action: La acción seleccionada
        """
        return self.policy.select_action(observation)
    
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
        
        # Inicializar entradas en la tabla Q si no existen
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0.0
        
        # Calcular Q máximo para el siguiente estado
        if done:
            max_next_q = 0
        else:
            # Obtener todos los valores Q para el siguiente estado
            next_q_values = [self.q_table.get((next_state, a), 0.0) for a in range(self.n_actions)]
            max_next_q = max(next_q_values)
        
        # Actualizar valor Q usando la ecuación de Bellman
        # Q(s,a) := Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
        current_q = self.q_table[(state, action)]
        td_target = reward + self.discount_factor * max_next_q
        td_error = td_target - current_q
        self.q_table[(state, action)] += self.learning_rate * td_error
        
        # Actualizar epsilon para la política
        self.policy.update()
    
    def reset(self):
        """
        Reinicia el estado interno del agente al comienzo de un episodio.
        """
        # Por lo general, no es necesario reiniciar nada para Q-Learning básico
        pass
    
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