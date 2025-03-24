#!/usr/bin/env python
# examples/tsp/tsp_example.py
"""
Ejemplo de uso del entorno TSP (Travelling Salesman Problem).
Este script demuestra cómo utilizar el entorno TSP con algoritmos de RL.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# Asegurar que el paquete principal esté en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Importar componentes de RLCore
from environments.tsp import TSPEnv
from agents.q_learning import QLearningAgent
from training.trainer import Trainer

def customize_observation_space_wrapper(env):
    """
    Wrapper para personalizar la forma en que se maneja el espacio de observación del TSP.
    """
    # Guardar la referencia al método original
    original_get_observation = env._get_observation
    
    # Definir nuevo método de observación
    def new_get_observation():
        # Obtener observación del formato original (diccionario)
        obs_dict = original_get_observation()
        
        # Convertir a formato que pueda ser usado como llave en tabla Q
        current_city = obs_dict['current_city']
        cities_visited_tuple = tuple(obs_dict['cities_visited'])
        
        # Retornar tupla (current_city, cities_visited) como observación
        return (current_city, cities_visited_tuple)
    
    # Reemplazar método
    env._get_observation = new_get_observation
    
    return env

def main():
    """Función principal del ejemplo."""
    print("Ejemplo del entorno TSP (Travelling Salesman Problem)")
    
    # Asegurar que la carpeta de resultados exista
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Parámetros
    n_cities = 5  # Usar un número pequeño para que Q-Learning sea factible
    n_episodes = 1000
    
    # Crear entorno TSP
    env = TSPEnv(n_cities=n_cities)
    
    # Aplicar wrapper para personalizar observaciones
    env = customize_observation_space_wrapper(env)
    
    # Crear agente Q-Learning
    agent = QLearningAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # Crear entrenador
    trainer = Trainer(agent, env, max_steps_per_episode=n_cities + 1)
    
    # Entrenar agente
    print("\nIniciando entrenamiento...")
    stats = trainer.train(
        n_episodes=n_episodes,
        render_every=None,
        verbose=True
    )
    
    # Visualizar curva de aprendizaje
    plt.figure(figsize=(10, 6))
    ax = trainer.plot_rewards(window=50, title="Curva de Aprendizaje: Q-Learning en TSP")
    
    # Guardar la figura en la carpeta de resultados
    learning_curve_path = os.path.join(results_dir, 'tsp_learning_curve.png')
    plt.savefig(learning_curve_path)
    plt.close()
    
    # Evaluar agente entrenado
    print("\nEvaluando agente entrenado...")
    eval_stats = trainer.evaluate(n_episodes=5, render=True)
    print(f"Recompensa media: {eval_stats['mean_reward']:.2f}")
    print(f"Longitud media de episodio: {eval_stats['mean_length']:.2f}")
    
    # Comparar con otras estrategias
    print("\nComparando con estrategias heurísticas...")
    
    # Implementación de estrategia Nearest Neighbor para TSP
    def nearest_neighbor_tsp():
        # Reiniciar entorno
        observation = env.reset()
        current_city, cities_visited = observation
        
        total_reward = 0
        done = False
        
        while not done:
            # Obtener distancias a todas las ciudades
            distances = env.distance_matrix[current_city]
            
            # Máscara para ciudades ya visitadas (distancia infinita)
            mask = np.array(cities_visited, dtype=bool)
            masked_distances = np.where(mask, np.inf, distances)
            
            # Seleccionar ciudad más cercana no visitada
            next_city = np.argmin(masked_distances)
            
            # Ejecutar acción
            observation, reward, done, info = env.step(next_city)
            
            # Actualizar estado
            if not done:
                current_city, cities_visited = observation
            
            total_reward += reward
        
        return total_reward, info['total_distance'], info['tour']
    
    # Implementación de estrategia aleatoria para TSP
    def random_tsp():
        # Reiniciar entorno
        observation = env.reset()
        current_city, cities_visited = observation
        
        total_reward = 0
        done = False
        
        while not done:
            # Obtener ciudades disponibles
            available_cities = [i for i in range(env.n_cities) if not cities_visited[i]]
            
            # Si hay ciudades disponibles, elegir una aleatoriamente
            if available_cities:
                next_city = np.random.choice(available_cities)
                
                # Ejecutar acción
                observation, reward, done, info = env.step(next_city)
                
                # Actualizar estado
                if not done:
                    current_city, cities_visited = observation
                
                total_reward += reward
            else:
                break
        
        return total_reward, info['total_distance'], info['tour']
    
    # Ejecutar estrategias
    print("\nEjecutando estrategia Nearest Neighbor...")
    nn_rewards = []
    nn_distances = []
    
    for _ in range(10):
        reward, distance, tour = nearest_neighbor_tsp()
        nn_rewards.append(reward)
        nn_distances.append(distance)
    
    print(f"  Recompensa media: {np.mean(nn_rewards):.2f}")
    print(f"  Distancia media: {np.mean(nn_distances):.2f}")
    
    print("\nEjecutando estrategia Aleatoria...")
    random_rewards = []
    random_distances = []
    
    for _ in range(10):
        reward, distance, tour = random_tsp()
        random_rewards.append(reward)
        random_distances.append(distance)
    
    print(f"  Recompensa media: {np.mean(random_rewards):.2f}")
    print(f"  Distancia media: {np.mean(random_distances):.2f}")
    
    # Comparar estrategias
    print("\nComparación final:")
    print(f"  Q-Learning: Recompensa = {eval_stats['mean_reward']:.2f}")
    print(f"  Nearest Neighbor: Recompensa = {np.mean(nn_rewards):.2f}")
    print(f"  Aleatorio: Recompensa = {np.mean(random_rewards):.2f}")
    
    # Visualizar solución final
    plt.figure(figsize=(8, 8))
    
    # Renderizar entorno final
    observation = env.reset()
    done = False
    
    # Seguir la política aprendida
    while not done:
        # Obtener acción según política sin exploración
        agent.policy.epsilon = 0
        current_city, cities_visited = observation
        action = agent.select_action(observation)
        
        # Ejecutar acción
        observation, reward, done, info = env.step(action)
    
    # Renderizar la solución
    env.render()
    
    # Guardar la visualización
    solution_path = os.path.join(results_dir, 'tsp_solution.png')
    plt.savefig(solution_path)
    plt.close()
    
    print(f"\nEjemplo completado. Se han guardado las visualizaciones en:")
    print(f"  - Curva de aprendizaje: '{learning_curve_path}'")
    print(f"  - Solución: '{solution_path}'")

if __name__ == "__main__":
    main()