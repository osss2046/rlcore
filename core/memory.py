# rlcore/core/memory.py
import numpy as np
from collections import deque
import random
from .experience import Experience

class ReplayBuffer:
    """
    Buffer para almacenar y muestrear experiencias.
    Fundamental para algoritmos off-policy como DQN.
    """
    
    def __init__(self, capacity):
        """
        Inicializa un buffer de repetición con capacidad fija.
        
        Args:
            capacity: Número máximo de experiencias a almacenar
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done, info=None):
        """
        Añade una experiencia al buffer.
        
        Args:
            state: Estado actual
            action: Acción tomada
            reward: Recompensa recibida
            next_state: Estado siguiente
            done: Indicador de fin de episodio
            info: Información adicional opcional
        """
        experience = Experience(state, action, reward, next_state, done, info)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        Muestrea un batch aleatorio de experiencias.
        
        Args:
            batch_size: Tamaño del batch a muestrear
            
        Returns:
            list: Lista de experiencias muestreadas
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        """Devuelve el número actual de experiencias en el buffer."""
        return len(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Buffer de repetición con muestreo prioritario basado en error TD.
    Mejora la eficiencia del aprendizaje al muestrear con más frecuencia
    experiencias con mayor error.
    """
    
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        Inicializa un buffer de repetición prioritario.
        
        Args:
            capacity: Capacidad máxima del buffer
            alpha: Factor de priorización (0 = sin priorización, 1 = priorización completa)
            beta: Factor de corrección de importance sampling (0 = sin corrección, 1 = corrección completa)
            beta_increment: Incremento de beta por cada muestreo
        """
        super().__init__(capacity)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        self.pos = 0
    
    def add(self, state, action, reward, next_state, done, info=None):
        """
        Añade una experiencia con prioridad máxima.
        
        Args:
            state: Estado actual
            action: Acción tomada
            reward: Recompensa recibida
            next_state: Estado siguiente
            done: Indicador de fin de episodio
            info: Información adicional opcional
        """
        experience = Experience(state, action, reward, next_state, done, info)
        
        if len(self.buffer) < self.buffer.maxlen:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
            
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.buffer.maxlen
    
    def sample(self, batch_size):
        """
        Muestrea un batch basado en prioridades.
        
        Args:
            batch_size: Tamaño del batch a muestrear
            
        Returns:
            tuple: (experiencias, índices, pesos)
        """
        if len(self.buffer) == 0:
            return []
            
        n_samples = min(batch_size, len(self.buffer))
        
        # Calcular probabilidades de muestreo
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Muestrear índices basados en probabilidades
        indices = np.random.choice(len(self.buffer), n_samples, p=probs, replace=False)
        
        # Calcular pesos para corrección de importance sampling
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        experiences = [self.buffer[idx] for idx in indices]
        
        return experiences, indices, weights
    
    def update_priorities(self, indices, priorities):
        """
        Actualiza las prioridades de las experiencias.
        
        Args:
            indices: Índices de las experiencias a actualizar
            priorities: Nuevas prioridades (típicamente basadas en error TD)
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)