# rlcore/agents/monte_carlo.py
import numpy as np
import pickle
from collections import defaultdict
from core.agent import Agent
from agents.q_learning import EpsilonGreedyPolicy

class MonteCarloAgent(Agent):
    """
    Agente que implementa el método de Monte Carlo para control.
    Usa muestreo de episodios completos para estimar valores de estado-acción
    y mejorar la política.
    """
    
    def __init__(self, action_space, observation_space, first_visit=True, discount_factor=0.99, 
                 epsilon=0.1, epsilon_decay=0.999, epsilon_min=0.01):
        """
        Inicializa el agente de Monte Carlo.
        
        Args:
            action_space: Espacio de acciones del entorno
            observation_space: Espacio de observaciones del entorno
            first_visit: Si True, usa "first-visit MC", si False usa "every-visit MC"
            discount_factor: Factor de descuento (gamma)
            epsilon: Probabilidad inicial de exploración
            epsilon_decay: Factor de decaimiento de epsilon
            epsilon_min: Valor mínimo de epsilon
        """
        super().__init__(action_space, observation_space)
        
        # Determinar si estamos tratando con espacios discretos
        self._check_spaces()
        
        # Hiperparámetros
        self.first_visit = first_visit
        self.discount_factor = discount_factor
        
        # Diccionarios para seguimiento de sumatorias y contadores de retornos
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(int)
        
        # Función Q (tabla)
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        
        # Política
        self.policy = EpsilonGreedyPolicy(
            action_space, 
            self.q_table, 
            epsilon, 
            epsilon_decay, 
            epsilon_min
        )
        
        # Almacenamiento de episodio actual
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
    
    def _check_spaces(self):
        """
        Verifica que los espacios de observación y acción sean compatibles con Monte Carlo.
        """
        # Para action_space
        if hasattr(self.action_space, 'n'):
            self.n_actions = self.action_space.n
        elif isinstance(self.action_space, int):
            self.n_actions = self.action_space
        else:
            raise ValueError("MonteCarloAgent requiere un espacio de acciones discreto")
    
    def select_action(self, observation):
        """
        Selecciona una acción basada en la observación actual.
        
        Args:
            observation: La observación actual del entorno
            
        Returns:
            action: La acción seleccionada
        """
        # Seleccionar acción según política
        action = self.policy.select_action(observation)
        
        # Almacenar estado y acción para el episodio actual
        self.episode_states.append(observation)
        self.episode_actions.append(action)
        
        return action
    
    def update(self, experience):
        """
        Guarda la experiencia para procesamiento al final del episodio.
        
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
        
        # Almacenar recompensa
        self.episode_rewards.append(reward)
        
        # Si es el final del episodio, actualizar valores Q
        if done:
            self._update_q_values()
    
    def _update_q_values(self):
        """
        Actualiza los valores Q usando el método de Monte Carlo.
        """
        # Verificar que tengamos datos almacenados
        if not self.episode_states:
            return
        
        # Generar estados-acciones únicos visitados si es first-visit MC
        # o usar todos los estados-acciones para every-visit MC
        if self.first_visit:
            # Mapea pares estado-acción a la primera ocurrencia
            state_action_first_visit = {}
            for idx, (state, action) in enumerate(zip(self.episode_states, self.episode_actions)):
                if (state, action) not in state_action_first_visit:
                    state_action_first_visit[(state, action)] = idx
            
            # Usar solo las primeras visitas para actualizar
            pairs_to_update = state_action_first_visit.items()
        else:
            # Usar todas las visitas para actualizar
            pairs_to_update = [((s, a), idx) for idx, (s, a) in 
                              enumerate(zip(self.episode_states, self.episode_actions))]
        
        # Calcular retornos para cada paso
        G = 0
        returns = []
        
        # Iterar en orden inverso para calcular retornos con descuento
        for r in reversed(self.episode_rewards):
            G = r + self.discount_factor * G
            returns.insert(0, G)
        
        # Actualizar valores Q
        for (state, action), idx in pairs_to_update:
            # Obtener retorno correspondiente a esta visita
            G = returns[idx]
            
            # Actualizar media de retornos
            self.returns_sum[(state, action)] += G
            self.returns_count[(state, action)] += 1
            self.q_table[state][action] = self.returns_sum[(state, action)] / self.returns_count[(state, action)]
        
        # Actualizar política (decaer epsilon)
        self.policy.update()
        
        # Limpiar datos del episodio para el próximo
        self._reset_episode()
    
    def _reset_episode(self):
        """
        Limpia los datos del episodio actual.
        """
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
    
    def reset(self):
        """
        Reinicia el estado interno del agente al comienzo de un episodio.
        """
        self._reset_episode()
    
    def save(self, filepath):
        """
        Guarda los valores Q y parámetros en un archivo.
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        # Convertir defaultdict a dict regular para serialización
        q_table_dict = {state: list(actions) for state, actions in self.q_table.items()}
        returns_sum_dict = dict(self.returns_sum)
        returns_count_dict = dict(self.returns_count)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': q_table_dict,
                'returns_sum': returns_sum_dict,
                'returns_count': returns_count_dict,
                'params': {
                    'first_visit': self.first_visit,
                    'discount_factor': self.discount_factor,
                    'epsilon': self.policy.epsilon,
                    'epsilon_decay': self.policy.epsilon_decay,
                    'epsilon_min': self.policy.epsilon_min
                }
            }, f)
    
    def load(self, filepath):
        """
        Carga valores Q y parámetros previamente guardados.
        
        Args:
            filepath: Ruta del modelo a cargar
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
            # Reconstruir defaultdict
            self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
            for state, actions in data['q_table'].items():
                self.q_table[state] = np.array(actions)
            
            self.returns_sum = defaultdict(float, data['returns_sum'])
            self.returns_count = defaultdict(int, data['returns_count'])
            
            self.first_visit = data['params']['first_visit']
            self.discount_factor = data['params']['discount_factor']
            self.policy.epsilon = data['params']['epsilon']
            self.policy.epsilon_decay = data['params']['epsilon_decay']
            self.policy.epsilon_min = data['params']['epsilon_min']