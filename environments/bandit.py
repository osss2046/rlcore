# rlcore/environments/bandit.py
import numpy as np
from core.environment import Environment
from core.space import Discrete

class MultiArmedBanditEnv(Environment):
    """
    Implementación del entorno de Bandit Multi-Brazo.
    Este entorno representa un conjunto de máquinas tragamonedas ("bandits") con diferentes
    distribuciones de recompensa. El agente debe aprender qué máquinas proporcionan
    mayor recompensa esperada.
    """
    
    def __init__(self, n_arms=10, reward_type='gaussian', stationary=True, initial_values=None,
                 reward_parameters=None, random_walk_sigma=0.01):
        """
        Inicializa el entorno de Bandit Multi-Brazo.
        
        Args:
            n_arms: Número de brazos (acciones) disponibles
            reward_type: Tipo de distribución de recompensa ('gaussian', 'bernoulli', 'exponential')
            stationary: Si es True, las distribuciones de recompensa no cambian con el tiempo
            initial_values: Valores iniciales para cada brazo (si es None, se generan aleatoriamente)
            reward_parameters: Parámetros específicos para las distribuciones de recompensa
                               (si es None, se generan aleatoriamente)
            random_walk_sigma: Desviación estándar para random walk en caso no estacionario
        """
        self.n_arms = n_arms
        self.reward_type = reward_type
        self.stationary = stationary
        self.random_walk_sigma = random_walk_sigma
        
        # Inicializar parámetros de recompensa
        if reward_parameters is None:
            if reward_type == 'gaussian':
                # Para distribución normal, usamos media y desviación estándar
                self.reward_means = np.random.normal(0, 1, n_arms)
                self.reward_stds = np.ones(n_arms)
            elif reward_type == 'bernoulli':
                # Para Bernoulli, usamos probabilidades de éxito
                self.reward_probs = np.random.uniform(0, 1, n_arms)
            elif reward_type == 'exponential':
                # Para exponencial, usamos lambda (tasa)
                self.reward_lambdas = np.random.uniform(0.5, 2.0, n_arms)
            else:
                raise ValueError(f"Tipo de recompensa no soportado: {reward_type}")
        else:
            # Usar parámetros proporcionados
            self.reward_parameters = reward_parameters
        
        # Valores verdaderos (desconocidos para el agente)
        if initial_values is None:
            if reward_type == 'gaussian':
                self.true_values = self.reward_means.copy()
            elif reward_type == 'bernoulli':
                self.true_values = self.reward_probs.copy()
            elif reward_type == 'exponential':
                self.true_values = 1.0 / self.reward_lambdas.copy()  # Valor esperado para exp(lambda) es 1/lambda
        else:
            self.true_values = initial_values.copy()
        
        # Estado actual (para cumplir con la interfaz de Environment)
        self.current_step = 0
        self.total_reward = 0
    
    def step(self, action):
        """
        Ejecuta una acción en el entorno (tira de un brazo específico).
        
        Args:
            action: Índice del brazo a tirar (entre 0 y n_arms-1)
            
        Returns:
            observation: Siempre 0 (el estado es irrelevante en bandits puros)
            reward: Recompensa obtenida
            done: False (el problema de bandit no termina)
            info: Información adicional
        """
        # Verificar que la acción sea válida
        assert 0 <= action < self.n_arms, f"Acción {action} fuera de rango [0, {self.n_arms-1}]"
        
        # Generar recompensa según el tipo de distribución
        if self.reward_type == 'gaussian':
            reward = np.random.normal(self.reward_means[action], self.reward_stds[action])
        elif self.reward_type == 'bernoulli':
            reward = 1.0 if np.random.random() < self.reward_probs[action] else 0.0
        elif self.reward_type == 'exponential':
            reward = np.random.exponential(1.0 / self.reward_lambdas[action])
        
        # Actualizar parámetros si no es estacionario
        if not self.stationary:
            self._update_parameters()
        
        # Actualizar estado
        self.current_step += 1
        self.total_reward += reward
        
        # En bandits puros, el estado es irrelevante (siempre 0)
        observation = 0
        done = False
        info = {
            'true_values': self.true_values.copy(),
            'optimal_arm': np.argmax(self.true_values),
            'regret': np.max(self.true_values) - self.true_values[action]
        }
        
        return observation, reward, done, info
    
    def reset(self):
        """
        Reinicia el entorno.
        
        Returns:
            observation: 0 (el estado es irrelevante en bandits puros)
        """
        self.current_step = 0
        self.total_reward = 0
        
        # Si no es estacionario, podríamos reiniciar los parámetros
        if not self.stationary and self.current_step > 0:
            self._reset_parameters()
        
        return 0
    
    def _update_parameters(self):
        """
        Actualiza los parámetros para el caso no estacionario usando random walk.
        """
        if self.reward_type == 'gaussian':
            # Actualizar medias con random walk
            self.reward_means += np.random.normal(0, self.random_walk_sigma, self.n_arms)
            self.true_values = self.reward_means.copy()
        elif self.reward_type == 'bernoulli':
            # Actualizar probabilidades con random walk, manteniéndolas en [0, 1]
            self.reward_probs += np.random.normal(0, self.random_walk_sigma, self.n_arms)
            self.reward_probs = np.clip(self.reward_probs, 0, 1)
            self.true_values = self.reward_probs.copy()
        elif self.reward_type == 'exponential':
            # Actualizar lambdas con random walk, manteniéndolas positivas
            self.reward_lambdas += np.random.normal(0, self.random_walk_sigma, self.n_arms)
            self.reward_lambdas = np.maximum(self.reward_lambdas, 0.1)  # Evitar valores negativos o muy cercanos a cero
            self.true_values = 1.0 / self.reward_lambdas.copy()
    
    def _reset_parameters(self):
        """
        Reinicia los parámetros a valores iniciales aleatorios.
        """
        if self.reward_type == 'gaussian':
            self.reward_means = np.random.normal(0, 1, self.n_arms)
            self.true_values = self.reward_means.copy()
        elif self.reward_type == 'bernoulli':
            self.reward_probs = np.random.uniform(0, 1, self.n_arms)
            self.true_values = self.reward_probs.copy()
        elif self.reward_type == 'exponential':
            self.reward_lambdas = np.random.uniform(0.5, 2.0, self.n_arms)
            self.true_values = 1.0 / self.reward_lambdas.copy()
    
    def render(self, mode='human'):
        """
        Renderiza el estado actual del entorno.
        
        Args:
            mode: Modo de renderizado
            
        Returns:
            None para 'human'
        """
        if mode == 'human':
            print(f"Paso: {self.current_step}")
            print(f"Recompensa total: {self.total_reward}")
            print("Valores verdaderos de los brazos:")
            for i, value in enumerate(self.true_values):
                print(f"  Brazo {i}: {value:.4f}")
            print(f"Brazo óptimo: {np.argmax(self.true_values)}")
            return None
        else:
            return None
    
    @property
    def action_space(self):
        """
        Define el espacio de acciones como los índices de los brazos.
        
        Returns:
            Discrete: Espacio de acciones discreto
        """
        return Discrete(self.n_arms)
    
    @property
    def observation_space(self):
        """
        En bandits puros, el espacio de observación es trivial (un solo estado).
        
        Returns:
            Discrete: Espacio de observaciones discreto con un solo estado
        """
        return Discrete(1)
    
    def seed(self, seed=None):
        """
        Establece la semilla para generación de números aleatorios.
        
        Args:
            seed: Semilla para el generador
            
        Returns:
            list: Lista de semillas utilizadas
        """
        if seed is not None:
            np.random.seed(seed)
        return [seed]