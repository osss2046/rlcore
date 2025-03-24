# rlcore/agents/dqn_pytorch.py
"""
Implementación de Deep Q-Network (DQN) usando PyTorch.
"""
import numpy as np
import random
import pickle
import os
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from core.agent import Agent
from core.memory import ReplayBuffer

class QNetwork(nn.Module):
    """
    Red neuronal para aproximar la función Q.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128]):
        """
        Inicializa la red Q.
        
        Args:
            state_dim: Dimensión del espacio de estados
            action_dim: Dimensión del espacio de acciones
            hidden_dims: Lista con el tamaño de las capas ocultas
        """
        super(QNetwork, self).__init__()
        
        # Construir capas
        layers = []
        prev_dim = state_dim
        
        # Capas ocultas
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Capa de salida
        layers.append(nn.Linear(prev_dim, action_dim))
        
        # Crear secuencia de capas
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Tensor de entrada (estado)
            
        Returns:
            Tensor de salida (valores Q para cada acción)
        """
        return self.model(x)


class DQNAgent(Agent):
    """
    Agente que implementa Deep Q-Network (DQN).
    """
    
    def __init__(self, action_space, observation_space, 
                 learning_rate=0.001, discount_factor=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64,
                 target_update_freq=10, hidden_dims=[128, 128],
                 use_double_dqn=False, use_dueling=False):
        """
        Inicializa el agente DQN.
        
        Args:
            action_space: Espacio de acciones del entorno
            observation_space: Espacio de observaciones del entorno
            learning_rate: Tasa de aprendizaje para el optimizador
            discount_factor: Factor de descuento para recompensas futuras
            epsilon_start: Valor inicial de epsilon para exploración
            epsilon_end: Valor mínimo de epsilon
            epsilon_decay: Factor de decaimiento de epsilon por cada actualización
            buffer_size: Tamaño del buffer de experiencias
            batch_size: Tamaño del batch para entrenamiento
            target_update_freq: Frecuencia de actualización de la red target (en episodios)
            hidden_dims: Dimensiones de las capas ocultas
            use_double_dqn: Si True, usa Double DQN
            use_dueling: Si True, usa arquitectura Dueling DQN
        """
        super().__init__(action_space, observation_space)
        
        # Hiperparámetros
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        self.use_dueling = use_dueling
        
        # Determinar dimensiones de estado y acción
        self.state_dim, self.action_dim = self._get_dimensions()
        
        # Inicializar redes (policy y target)
        self.policy_net = QNetwork(self.state_dim, self.action_dim, hidden_dims)
        self.target_net = QNetwork(self.state_dim, self.action_dim, hidden_dims)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network en modo evaluación
        
        # Optimizador
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Buffer de experiencias
        self.memory = ReplayBuffer(capacity=buffer_size)
        
        # Contadores
        self.episode_count = 0
        self.steps_done = 0
        
        # Dispositivo (CPU/GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
    
    def _get_dimensions(self):
        """
        Determina las dimensiones del espacio de estados y acciones.
        
        Returns:
            tuple: (state_dim, action_dim)
        """
        # Determinar dimensión de acción
        if hasattr(self.action_space, 'n'):
            action_dim = self.action_space.n
        else:
            action_dim = self.action_space
            
        # Determinar dimensión de estado
        if hasattr(self.observation_space, 'shape'):
            state_dim = int(np.prod(self.observation_space.shape))
        elif hasattr(self.observation_space, 'n'):
            state_dim = self.observation_space.n
        else:
            state_dim = self.observation_space
            
        return state_dim, action_dim
    
    def _preprocess_state(self, state):
        """
        Preprocesa el estado para la red.
        
        Args:
            state: Estado del entorno
            
        Returns:
            torch.Tensor: Tensor de estado preprocesado
        """
        # Convertir a array si no lo es
        if not isinstance(state, np.ndarray):
            if isinstance(state, (int, float)):
                # Estados discretos: codificación one-hot
                processed = np.zeros(self.state_dim)
                processed[state] = 1
            else:
                # Intentar convertir a array
                processed = np.array(state)
        else:
            processed = state.copy()
        
        # Aplanar si es necesario
        if processed.shape and len(processed.shape) > 1:
            processed = processed.flatten()
        
        # Asegurar el tipo float32
        if not isinstance(processed, float):
            processed = processed.astype(np.float32)
        
        # Convertir a tensor
        return torch.FloatTensor(processed).to(self.device)
    
    def select_action(self, state):
        """
        Selecciona una acción usando política epsilon-greedy.
        
        Args:
            state: Estado actual del entorno
            
        Returns:
            int: Acción seleccionada
        """
        # Incrementar contador de pasos
        self.steps_done += 1
        
        # Exploración: con probabilidad epsilon, seleccionar acción aleatoria
        if random.random() < self.epsilon:
            if hasattr(self.action_space, 'sample'):
                return self.action_space.sample()
            else:
                return random.randint(0, self.action_dim - 1)
        
        # Explotación: seleccionar mejor acción según la red
        with torch.no_grad():
            state_tensor = self._preprocess_state(state)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def update(self, experience):
        """
        Actualiza el agente con una nueva experiencia.
        
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
        
        # Guardar en buffer de experiencias
        self.memory.add(state, action, reward, next_state, done)
        
        # Solo entrenar si hay suficientes experiencias
        if len(self.memory) < self.batch_size:
            return
        
        # Muestrear batch de experiencias
        batch = self.memory.sample(self.batch_size)
        
        # Preparar batch para entrenamiento
        state_batch = torch.cat([self._preprocess_state(exp.state).unsqueeze(0) for exp in batch])
        action_batch = torch.tensor([[exp.action] for exp in batch], dtype=torch.int64).to(self.device)
        reward_batch = torch.tensor([[exp.reward] for exp in batch], dtype=torch.float32).to(self.device)
        next_state_batch = torch.cat([self._preprocess_state(exp.next_state).unsqueeze(0) for exp in batch])
        done_batch = torch.tensor([[exp.done] for exp in batch], dtype=torch.float32).to(self.device)
        
        # Calcular valores Q para el estado actual
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Calcular valores Q esperados
        if self.use_double_dqn:
            # Double DQN: seleccionar acciones con policy_net, evaluar con target_net
            next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
        else:
            # DQN estándar: usar target_net para evaluar
            next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
        
        # Calcular valores objetivo
        target_q_values = reward_batch + (1 - done_batch) * self.discount_factor * next_q_values
        
        # Calcular pérdida
        loss = F.smooth_l1_loss(q_values, target_q_values)
        
        # Optimizar modelo
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)  # Gradiente clipping
        self.optimizer.step()
    
    def end_episode(self):
        """
        Realiza tareas necesarias al final de un episodio.
        """
        # Decaer epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Incrementar contador de episodios
        self.episode_count += 1
        
        # Actualizar red target si corresponde
        if self.episode_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def reset(self):
        """
        Reinicia el estado interno del agente al comienzo de un episodio.
        """
        pass
    
    def save(self, filepath):
        """
        Guarda el modelo y parámetros en un archivo.
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        model_state = {
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'params': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq,
                'use_double_dqn': self.use_double_dqn,
                'use_dueling': self.use_dueling
            }
        }
        
        torch.save(model_state, filepath)
    
    def load(self, filepath):
        """
        Carga un modelo previamente guardado.
        
        Args:
            filepath: Ruta del modelo a cargar
        """
        model_state = torch.load(filepath)
        
        # Recrear redes si es necesario
        if hasattr(model_state['params'], 'state_dim'):
            self.state_dim = model_state['params']['state_dim']
            self.action_dim = model_state['params']['action_dim']
            
            # Recrear redes solo si las dimensiones son diferentes
            if (self.policy_net.model[0].in_features != self.state_dim or 
                self.policy_net.model[-1].out_features != self.action_dim):
                self.policy_net = QNetwork(self.state_dim, self.action_dim)
                self.target_net = QNetwork(self.state_dim, self.action_dim)
        
        # Cargar parámetros
        self.policy_net.load_state_dict(model_state['policy_state_dict'])
        self.target_net.load_state_dict(model_state['target_state_dict'])
        self.optimizer.load_state_dict(model_state['optimizer_state_dict'])
        
        # Cargar hiperparámetros
        for key, value in model_state['params'].items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Mover modelos al dispositivo correcto
        self.policy_net.to(self.device)
        self.target_net.to(self.device)


class DuelingQNetwork(nn.Module):
    """
    Red neuronal con arquitectura Dueling para aproximar la función Q.
    Separa la estimación de valores de estado y ventajas de acciones.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128]):
        """
        Inicializa la red Q Dueling.
        
        Args:
            state_dim: Dimensión del espacio de estados
            action_dim: Dimensión del espacio de acciones
            hidden_dims: Lista con el tamaño de las capas ocultas
        """
        super(DuelingQNetwork, self).__init__()
        
        # Capas compartidas
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU()
        )
        
        # Rama de valor (V)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )
        
        # Rama de ventaja (A)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Tensor de entrada (estado)
            
        Returns:
            Tensor de salida (valores Q para cada acción)
        """
        shared = self.shared_layers(x)
        
        value = self.value_stream(shared)
        advantages = self.advantage_stream(shared)
        
        # Combinar valor y ventajas: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        return value + (advantages - advantages.mean(dim=1, keepdim=True))


class DuelingDQNAgent(DQNAgent):
    """
    Agente que implementa Dueling DQN.
    """
    
    def __init__(self, action_space, observation_space, 
                 learning_rate=0.001, discount_factor=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64,
                 target_update_freq=10, hidden_dims=[128, 128],
                 use_double_dqn=False):
        """
        Inicializa el agente Dueling DQN.
        """
        # Llamar al constructor de la clase padre pero sin inicializar las redes
        Agent.__init__(self, action_space, observation_space)
        
        # Hiperparámetros
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        self.use_dueling = True
        
        # Determinar dimensiones de estado y acción
        self.state_dim, self.action_dim = self._get_dimensions()
        
        # Inicializar redes dueling (policy y target)
        self.policy_net = DuelingQNetwork(self.state_dim, self.action_dim, hidden_dims)
        self.target_net = DuelingQNetwork(self.state_dim, self.action_dim, hidden_dims)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network en modo evaluación
        
        # Optimizador
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Buffer de experiencias
        self.memory = ReplayBuffer(capacity=buffer_size)
        
        # Contadores
        self.episode_count = 0
        self.steps_done = 0
        
        # Dispositivo (CPU/GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)