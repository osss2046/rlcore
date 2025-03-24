# rlcore/environments/gridworld.py
import numpy as np
from core.environment import Environment
from core.space import Discrete

class GridWorldEnv(Environment):
    """
    Implementación de un entorno GridWorld simple.
    
    El agente comienza en una posición inicial y debe navegar hasta la meta,
    evitando obstáculos. Recibe recompensa positiva por llegar a la meta,
    recompensa negativa por chocar con obstáculos o salirse de la cuadrícula.
    """
    
    def __init__(self, width=5, height=5, start_pos=(0, 0), goal_pos=(4, 4), 
                 obstacles=None, obstacle_reward=-1.0, goal_reward=1.0, 
                 step_reward=-0.01, max_steps=100):
        """
        Inicializa el entorno GridWorld.
        
        Args:
            width: Ancho de la cuadrícula
            height: Alto de la cuadrícula
            start_pos: Posición inicial (fila, columna)
            goal_pos: Posición de la meta (fila, columna)
            obstacles: Lista de posiciones con obstáculos [(fila, columna), ...]
            obstacle_reward: Recompensa por chocar con un obstáculo
            goal_reward: Recompensa por llegar a la meta
            step_reward: Recompensa por cada paso (para fomentar rutas cortas)
            max_steps: Número máximo de pasos antes de terminar el episodio
        """
        self.width = width
        self.height = height
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.obstacles = obstacles if obstacles is not None else []
        self.obstacle_reward = obstacle_reward
        self.goal_reward = goal_reward
        self.step_reward = step_reward
        self.max_steps = max_steps
        
        # Estado actual del agente
        self.agent_pos = None
        self.steps = 0
        
        # Definir acciones: 0=arriba, 1=derecha, 2=abajo, 3=izquierda
        self._action_to_dir = [
            (-1, 0),  # Arriba
            (0, 1),   # Derecha
            (1, 0),   # Abajo
            (0, -1)   # Izquierda
        ]
    
    @property
    def action_space(self):
        """
        Define el espacio de acciones: {0, 1, 2, 3}.
        
        Returns:
            Discrete: Espacio de acciones discreto
        """
        return Discrete(4)
    
    @property
    def observation_space(self):
        """
        Define el espacio de observaciones como tuplas (fila, columna).
        Para este entorno simple, la observación es la posición del agente.
        
        Returns:
            Discrete: Espacio de observaciones discreto
        """
        return self.width * self.height
    
    def reset(self):
        """
        Reinicia el entorno y devuelve la observación inicial.
        
        Returns:
            tuple: Posición inicial del agente como (fila, columna)
        """
        self.agent_pos = self.start_pos
        self.steps = 0
        return self._get_state()
    
    def step(self, action):
        """
        Ejecuta una acción en el entorno.
        
        Args:
            action: Acción a ejecutar (0=arriba, 1=derecha, 2=abajo, 3=izquierda)
            
        Returns:
            tuple: (nuevo_estado, recompensa, episodio_terminado, info)
        """
        # Incrementar contador de pasos
        self.steps += 1
        
        # Calcular nueva posición
        dr, dc = self._action_to_dir[action]
        new_pos = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
        
        # Verificar límites de la cuadrícula
        if not (0 <= new_pos[0] < self.height and 0 <= new_pos[1] < self.width):
            # El agente intentó salirse de la cuadrícula, no se mueve
            reward = self.obstacle_reward
            done = False
            info = {"message": "Intento salir de la cuadrícula"}
        # Verificar obstáculos
        elif new_pos in self.obstacles:
            # El agente chocó con un obstáculo, no se mueve
            reward = self.obstacle_reward
            done = False
            info = {"message": "Chocó con obstáculo"}
        else:
            # Movimiento válido, actualizar posición
            self.agent_pos = new_pos
            
            # Verificar si llegó a la meta
            if self.agent_pos == self.goal_pos:
                reward = self.goal_reward
                done = True
                info = {"message": "Llegó a la meta"}
            else:
                # Recompensa por paso (generalmente negativa)
                reward = self.step_reward
                done = False
                info = {"message": "Movimiento válido"}
        
        # Verificar si se alcanzó el límite de pasos
        if self.steps >= self.max_steps and not done:
            done = True
            info["message"] += ", se alcanzó el límite de pasos"
        
        return self._get_state(), reward, done, info
    
    def render(self, mode='human'):
        """
        Renderiza el estado actual del entorno.
        
        Args:
            mode: Modo de renderizado ('human' o 'rgb_array')
            
        Returns:
            None para 'human', numpy.ndarray para 'rgb_array'
        """
        if mode == 'human':
            grid = [['·' for _ in range(self.width)] for _ in range(self.height)]
            
            # Colocar obstáculos
            for obs in self.obstacles:
                grid[obs[0]][obs[1]] = 'X'
            
            # Colocar meta
            grid[self.goal_pos[0]][self.goal_pos[1]] = 'G'
            
            # Colocar agente
            grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
            
            # Imprimir cuadrícula
            print('\n'.join(' '.join(row) for row in grid))
            print(f"Posición del agente: {self.agent_pos}")
            print(f"Pasos: {self.steps}/{self.max_steps}")
            return None
        else:
            # Para otros modos de renderizado (no implementados)
            return None
    
    def _get_state(self):
        """
        Convierte la posición del agente en un estado único.
        
        Returns:
            int: Estado único representando la posición del agente
        """
        # Convertir coordenadas 2D en un índice unidimensional
        return self.agent_pos[0] * self.width + self.agent_pos[1]
    
    def _get_position(self, state):
        """
        Convierte un estado único en coordenadas 2D.
        
        Args:
            state: Estado único
            
        Returns:
            tuple: Posición (fila, columna)
        """
        row = state // self.width
        col = state % self.width
        return (row, col)