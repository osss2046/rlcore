# rlcore/environments/tsp.py
import numpy as np
import matplotlib.pyplot as plt
from core.environment import Environment
from core.space import Discrete, Box

class TSPEnv(Environment):
    """
    Entorno para el Problema del Viajante (TSP, Travelling Salesman Problem).
    El agente debe visitar todas las ciudades exactamente una vez y regresar
    a la ciudad inicial, minimizando la distancia total recorrida.
    """
    
    def __init__(self, n_cities=10, max_steps=None, use_coordinates=True, 
                 city_coordinates=None, distance_matrix=None, start_city=0):
        """
        Inicializa el entorno TSP.
        
        Args:
            n_cities: Número de ciudades
            max_steps: Número máximo de pasos (si None, será n_cities)
            use_coordinates: Si True, usa coordenadas 2D; si False, usa matriz de distancias
            city_coordinates: Coordenadas de las ciudades (si es None, se generan aleatoriamente)
            distance_matrix: Matriz de distancias entre ciudades (ignorada si use_coordinates=True)
            start_city: Ciudad inicial (por defecto, 0)
        """
        self.n_cities = n_cities
        self.max_steps = max_steps if max_steps is not None else n_cities
        self.use_coordinates = use_coordinates
        self.start_city = start_city
        
        # Inicializar coordenadas o matriz de distancias
        if use_coordinates:
            if city_coordinates is None:
                # Generar coordenadas aleatorias en [0, 1] x [0, 1]
                self.city_coordinates = np.random.random((n_cities, 2))
            else:
                self.city_coordinates = np.array(city_coordinates)
            
            # Calcular matriz de distancias
            self.distance_matrix = np.zeros((n_cities, n_cities))
            for i in range(n_cities):
                for j in range(i+1, n_cities):
                    # Distancia euclidiana
                    dist = np.linalg.norm(self.city_coordinates[i] - self.city_coordinates[j])
                    self.distance_matrix[i, j] = dist
                    self.distance_matrix[j, i] = dist
        else:
            if distance_matrix is None:
                raise ValueError("Si use_coordinates=False, debe proporcionar distance_matrix")
            self.distance_matrix = np.array(distance_matrix)
            self.city_coordinates = None
        
        # Estado interno
        self.current_city = None
        self.cities_visited = None
        self.tour = None
        self.step_count = None
        self.total_distance = None
        
        # Comprobar validez de la matriz de distancias
        assert self.distance_matrix.shape == (n_cities, n_cities), \
            f"La matriz de distancias debe ser {n_cities}x{n_cities}"
        
        # Calcular la mejor solución conocida usando fuerza bruta (sólo para n_cities pequeño)
        self.optimal_tour = None
        self.optimal_distance = float('inf')
        if n_cities <= 10:  # Limite para cálculo por fuerza bruta
            self._compute_optimal_tour()
    
    def reset(self):
        """
        Reinicia el entorno al estado inicial.
        
        Returns:
            observation: Representación del estado inicial
        """
        self.current_city = self.start_city
        self.cities_visited = np.zeros(self.n_cities, dtype=bool)
        self.cities_visited[self.current_city] = True
        self.tour = [self.current_city]
        self.step_count = 0
        self.total_distance = 0.0
        
        return self._get_observation()
    
    def step(self, action):
        """
        Ejecuta una acción en el entorno, visitando una nueva ciudad.
        
        Args:
            action: Índice de la ciudad a visitar
            
        Returns:
            observation: Nueva observación
            reward: Recompensa recibida
            done: Si el episodio ha terminado
            info: Información adicional
        """
        # Verificar validez de la acción
        if action < 0 or action >= self.n_cities:
            raise ValueError(f"Acción {action} fuera de rango [0, {self.n_cities-1}]")
        
        # Si la ciudad ya fue visitada, dar recompensa negativa y no cambiar el estado
        if self.cities_visited[action]:
            return self._get_observation(), -1.0, False, {"error": "Ciudad ya visitada"}
        
        # Calcular distancia
        distance = self.distance_matrix[self.current_city, action]
        
        # Actualizar estado
        self.current_city = action
        self.cities_visited[action] = True
        self.tour.append(action)
        self.total_distance += distance
        self.step_count += 1
        
        # Calcular recompensa (negativa de la distancia)
        reward = -distance
        
        # Verificar si se han visitado todas las ciudades o se alcanzó el límite de pasos
        all_visited = np.all(self.cities_visited)
        max_steps_reached = self.step_count >= self.max_steps
        
        # Si se han visitado todas las ciudades o se alcanzó el límite de pasos, terminar episodio
        done = all_visited or max_steps_reached
        
        # Si es el último paso y se visitaron todas las ciudades, agregar distancia de regreso
        if done and all_visited:
            return_distance = self.distance_matrix[self.current_city, self.start_city]
            self.total_distance += return_distance
            reward -= return_distance  # Incluir distancia de regreso en la recompensa
            self.tour.append(self.start_city)  # Agregar regreso a la ciudad inicial
        
        # Preparar info
        info = {
            "current_city": self.current_city,
            "cities_visited": self.cities_visited.copy(),
            "tour": self.tour.copy(),
            "total_distance": self.total_distance
        }
        
        if done:
            info["complete_tour"] = all_visited
            
            # Si conocemos la solución óptima, incluir comparación
            if self.optimal_tour is not None:
                info["optimal_tour"] = self.optimal_tour.copy()
                info["optimal_distance"] = self.optimal_distance
                info["optimality_gap"] = (self.total_distance - self.optimal_distance) / self.optimal_distance
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        """
        Crea la representación del estado actual.
        
        Returns:
            dict: Observación que contiene información sobre el estado actual
        """
        observation = {
            "current_city": self.current_city,
            "cities_visited": self.cities_visited.copy(),
            "distance_matrix": self.distance_matrix
        }
        return observation
    
    def _compute_optimal_tour(self):
        """
        Calcula la solución óptima mediante fuerza bruta.
        Solo factible para un número pequeño de ciudades.
        """
        from itertools import permutations
        
        # Generar todas las permutaciones posibles (excluyendo la ciudad inicial)
        other_cities = list(range(self.n_cities))
        other_cities.remove(self.start_city)
        
        best_tour = None
        best_distance = float('inf')
        
        for perm in permutations(other_cities):
            # Construir tour completo
            tour = [self.start_city] + list(perm) + [self.start_city]
            
            # Calcular distancia total
            distance = 0
            for i in range(len(tour) - 1):
                distance += self.distance_matrix[tour[i], tour[i+1]]
            
            # Actualizar mejor tour
            if distance < best_distance:
                best_distance = distance
                best_tour = tour
        
        self.optimal_tour = best_tour
        self.optimal_distance = best_distance
        
        print(f"Tour óptimo calculado: {best_tour}")
        print(f"Distancia óptima: {best_distance:.4f}")
    
    def render(self, mode='human'):
        """
        Renderiza el estado actual del TSP.
        
        Args:
            mode: Modo de renderizado ('human' o 'rgb_array')
            
        Returns:
            None para 'human', numpy.ndarray para 'rgb_array'
        """
        if mode == 'human':
            if self.city_coordinates is None:
                print("No se pueden renderizar las ciudades sin coordenadas.")
                return None
            
            plt.figure(figsize=(8, 8))
            
            # Dibujar ciudades
            plt.scatter(self.city_coordinates[:, 0], self.city_coordinates[:, 1], 
                        c='blue', s=100, label='Ciudades')
            
            # Resaltar ciudad actual
            plt.scatter(self.city_coordinates[self.current_city, 0], 
                        self.city_coordinates[self.current_city, 1],
                        c='red', s=150, label='Ciudad Actual')
            
            # Dibujar tour hasta ahora
            tour_coords = self.city_coordinates[self.tour]
            plt.plot(tour_coords[:, 0], tour_coords[:, 1], 'b-', alpha=0.5)
            
            # Anotar ciudades
            for i, (x, y) in enumerate(self.city_coordinates):
                plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
            
            plt.title(f"TSP - Paso {self.step_count}, Distancia {self.total_distance:.4f}")
            plt.legend()
            plt.grid(True)
            plt.xlim(-0.1, 1.1)
            plt.ylim(-0.1, 1.1)
            
            plt.show()
            return None
        else:
            return None
    
    @property
    def action_space(self):
        """
        Define el espacio de acciones como los índices de las ciudades.
        
        Returns:
            Discrete: Espacio de acciones discreto
        """
        return Discrete(self.n_cities)
    
    @property
    def observation_space(self):
        """
        Define el espacio de observaciones.
        Debido a la complejidad del estado, aquí lo representamos de forma simplificada.
        
        Returns:
            Discrete: Espacio de observaciones
        """
        # Esta es una simplificación, en realidad el espacio de observaciones es más complejo
        # y sería mejor representarlo como un Dict space, pero para compatibilidad con los
        # algoritmos actuales, lo representamos como un espacio discreto grande
        return self.n_cities
    
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