# rlcore/agents/policy_iteration.py
import numpy as np
import pickle
from core.agent import Agent

class PolicyIterationAgent(Agent):
    """
    Agente que implementa el algoritmo de Iteración de Política.
    Este método alterna entre evaluación de política y mejora de política
    hasta converger a la política óptima.
    Requiere conocimiento completo del modelo del entorno (P(s'|s,a) y R(s,a,s')).
    """
    
    def __init__(self, action_space, observation_space, model, discount_factor=0.99, 
                 theta=1e-6, max_iterations=1000):
        """
        Inicializa el agente de Iteración de Política.
        
        Args:
            action_space: Espacio de acciones del entorno
            observation_space: Espacio de observaciones del entorno
            model: Función que toma (estado, acción) y devuelve [(prob, estado_siguiente, recompensa, terminado)]
            discount_factor: Factor de descuento (gamma)
            theta: Umbral para convergencia
            max_iterations: Número máximo de iteraciones permitidas
        """
        super().__init__(action_space, observation_space)
        
        # Almacenar modelo del entorno
        self.model = model
        
        # Determinar si estamos tratando con espacios discretos
        self._check_spaces()
        
        # Hiperparámetros
        self.discount_factor = discount_factor
        self.theta = theta
        self.max_iterations = max_iterations
        
        # Inicializar función de valor y política
        self.value_function = np.zeros(self.n_states)
        self.policy = np.zeros(self.n_states, dtype=int)
        
        # Bandera para indicar si se ha ejecutado el algoritmo
        self.is_trained = False
    
    def _check_spaces(self):
        """
        Verifica que los espacios de observación y acción sean compatibles con Iteración de Política.
        """
        # Para action_space
        if hasattr(self.action_space, 'n'):
            self.n_actions = self.action_space.n
        elif isinstance(self.action_space, int):
            self.n_actions = self.action_space
        else:
            raise ValueError("PolicyIterationAgent requiere un espacio de acciones discreto")
        
        # Para observation_space
        if hasattr(self.observation_space, 'n'):
            self.n_states = self.observation_space.n
        elif isinstance(self.observation_space, int):
            self.n_states = self.observation_space
        else:
            raise ValueError("PolicyIterationAgent requiere un espacio de estados discreto (enumerable)")
    
    def _policy_evaluation(self, policy, theta=0.00001):
        """
        Evalúa una política calculando la función de valor estado.
        
        Args:
            policy: La política a evaluar
            theta: Umbral para convergencia
            
        Returns:
            np.array: Función de valor para la política dada
        """
        V = np.zeros(self.n_states)
        
        while True:
            delta = 0
            
            for s in range(self.n_states):
                v = V[s]
                
                # Acción actual para este estado según la política
                a = policy[s]
                
                # Obtener valor esperado
                v_new = 0
                
                # Obtener transiciones desde el modelo
                transitions = self.model(s, a)
                
                for prob, next_state, reward, done in transitions:
                    if done:
                        v_new += prob * reward
                    else:
                        v_new += prob * (reward + self.discount_factor * V[next_state])
                
                # Actualizar función de valor
                V[s] = v_new
                
                # Seguimiento de convergencia
                delta = max(delta, abs(v - V[s]))
            
            # Verificar convergencia
            if delta < theta:
                break
                
        return V
    
    def train(self, verbose=False):
        """
        Ejecuta el algoritmo de Iteración de Política hasta convergencia o máximo de iteraciones.
        
        Args:
            verbose: Si True, imprime información de progreso
            
        Returns:
            tuple: (policy, value_function, n_iterations)
        """
        # Inicializar política aleatoriamente
        policy = np.random.randint(0, self.n_actions, size=self.n_states)
        
        for i in range(self.max_iterations):
            # 1. Evaluación de política
            V = self._policy_evaluation(policy, theta=self.theta)
            
            # Bandera para seguimiento de convergencia
            policy_stable = True
            
            # 2. Mejora de política
            for s in range(self.n_states):
                # Acción actual para este estado
                old_action = policy[s]
                
                # Valores Q para cada acción
                q_values = np.zeros(self.n_actions)
                
                for a in range(self.n_actions):
                    # Obtener transiciones desde el modelo
                    transitions = self.model(s, a)
                    
                    for prob, next_state, reward, done in transitions:
                        if done:
                            q_values[a] += prob * reward
                        else:
                            q_values[a] += prob * (reward + self.discount_factor * V[next_state])
                
                # Elegir mejor acción
                best_action = np.argmax(q_values)
                
                # Actualizar política
                policy[s] = best_action
                
                # Verificar si la política cambió
                if old_action != best_action:
                    policy_stable = False
            
            # Informar progreso si se solicita
            if verbose:
                print(f"Iteración {i+1}: {'Política estable' if policy_stable else 'Política actualizada'}")
            
            # Verificar convergencia
            if policy_stable:
                if verbose:
                    print(f"Política Iteración convergió en {i+1} iteraciones.")
                break
        
        # Almacenar resultados
        self.value_function = V
        self.policy = policy
        self.is_trained = True
        
        return policy, V, i + 1
    
    def select_action(self, observation):
        """
        Selecciona una acción basada en la política derivada.
        
        Args:
            observation: El estado actual del entorno
            
        Returns:
            action: La acción según la política óptima
        """
        if not self.is_trained:
            raise RuntimeError("PolicyIterationAgent debe ser entrenado antes de seleccionar acciones.")
        
        return self.policy[observation]
    
    def update(self, experience):
        """
        No implementado para PolicyIterationAgent, ya que es un método de planificación
        que requiere entrenamiento previo.
        
        Args:
            experience: Objeto Experience o tupla (state, action, reward, next_state, done)
        """
        # La Iteración de Política es un método de planificación, no un método de aprendizaje
        # por refuerzo, por lo que no se actualiza con experiencias.
        pass
    
    def reset(self):
        """
        No es necesario para este agente, ya que no mantiene un estado interno durante ejecución.
        """
        pass
    
    def save(self, filepath):
        """
        Guarda la función de valor y la política en un archivo.
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'value_function': self.value_function,
                'policy': self.policy,
                'params': {
                    'discount_factor': self.discount_factor,
                    'theta': self.theta
                }
            }, f)
    
    def load(self, filepath):
        """
        Carga una función de valor y política previamente guardadas.
        
        Args:
            filepath: Ruta del modelo a cargar
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.value_function = data['value_function']
            self.policy = data['policy']
            self.discount_factor = data['params']['discount_factor']
            self.theta = data['params']['theta']
            self.is_trained = True