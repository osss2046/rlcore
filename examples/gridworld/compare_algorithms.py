#!/usr/bin/env python
# examples/gridworld/compare_algorithms.py
"""
Comparación de diferentes algoritmos de RL en un entorno GridWorld.
Este script compara Q-Learning, SARSA y Monte Carlo en términos de 
velocidad de convergencia y calidad de la política resultante.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# Asegurar que el paquete principal esté en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Importar componentes de RLCore
from environments.gridworld import GridWorldEnv
from agents.q_learning import QLearningAgent
from agents.sarsa import SARSAAgent
from agents.monte_carlo import MonteCarloAgent
from agents.value_iteration import ValueIterationAgent
from agents.policy_iteration import PolicyIterationAgent
from training.trainer import Trainer
from utils.visualization import plot_learning_curve, plot_value_function, plot_policy_arrows

def create_model_for_dp(env):
    """
    Crea un modelo del entorno para algoritmos de programación dinámica.
    
    Args:
        env: Entorno GridWorld
        
    Returns:
        function: Función modelo que toma (estado, acción) y devuelve transiciones
    """
    def model(state, action):
        """
        Modelo del entorno para programación dinámica.
        
        Args:
            state: Estado actual
            action: Acción a ejecutar
            
        Returns:
            list: Lista de tuplas (probabilidad, siguiente_estado, recompensa, terminado)
        """
        # Convertir estado a posición
        pos = env._get_position(state)
        row, col = pos
        
        # Obtener dirección de la acción
        dr, dc = env._action_to_dir[action]
        new_pos = (row + dr, col + dc)
        
        # Verificar si es un movimiento válido
        if not (0 <= new_pos[0] < env.height and 0 <= new_pos[1] < env.width):
            # Intento salir de la cuadrícula
            next_state = state  # No se mueve
            reward = env.obstacle_reward
            done = False
        elif new_pos in env.obstacles:
            # Chocó con un obstáculo
            next_state = state  # No se mueve
            reward = env.obstacle_reward
            done = False
        else:
            # Movimiento válido
            # Calcular el estado siguiente manualmente (sin usar _get_state)
            next_state = new_pos[0] * env.width + new_pos[1]
            
            if new_pos == env.goal_pos:
                # Llegó a la meta
                reward = env.goal_reward
                done = True
            else:
                # Paso normal
                reward = env.step_reward
                done = False
        
        # Devolver transición determinista con probabilidad 1.0
        return [(1.0, next_state, reward, done)]
    
    return model

def main():
    """Función principal del ejemplo de comparación."""
    print("Comparación de Algoritmos RL en GridWorld")
    
    # Asegurar que la carpeta de resultados exista
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Parámetros de experimento
    n_episodes = 500
    n_runs = 5  # Número de ejecuciones para promediar resultados
    
    # Crear entorno GridWorld
    obstacles = [(1, 1), (1, 2), (2, 1), (3, 3)]
    env = GridWorldEnv(
        width=5,
        height=5,
        start_pos=(0, 0),
        goal_pos=(4, 4),
        obstacles=obstacles,
        obstacle_reward=-0.5,
        goal_reward=1.0,
        step_reward=-0.1,
        max_steps=100
    )
    
    # Resultados para almacenar recompensas
    all_rewards = {
        'Q-Learning': [],
        'SARSA': [],
        'Monte Carlo': []
    }
    
    # Medir tiempo de ejecución
    execution_times = {}
    
    print("\nEntrenando algoritmos...")
    
    # Entrenar cada algoritmo varias veces y promediar resultados
    for algorithm in ['Q-Learning', 'SARSA', 'Monte Carlo']:
        start_time = time.time()
        
        for run in range(n_runs):
            print(f"Ejecutando {algorithm}, ejecución {run+1}/{n_runs}")
            
            # Reiniciar entorno
            env.reset()
            
            # Crear el agente apropiado
            if algorithm == 'Q-Learning':
                agent = QLearningAgent(
                    action_space=env.action_space,
                    observation_space=env.observation_space,
                    learning_rate=0.1,
                    discount_factor=0.99,
                    epsilon=1.0,
                    epsilon_decay=0.995,
                    epsilon_min=0.01
                )
            elif algorithm == 'SARSA':
                agent = SARSAAgent(
                    action_space=env.action_space,
                    observation_space=env.observation_space,
                    learning_rate=0.1,
                    discount_factor=0.99,
                    epsilon=1.0,
                    epsilon_decay=0.995,
                    epsilon_min=0.01
                )
            elif algorithm == 'Monte Carlo':
                agent = MonteCarloAgent(
                    action_space=env.action_space,
                    observation_space=env.observation_space,
                    first_visit=True,
                    discount_factor=0.99,
                    epsilon=1.0,
                    epsilon_decay=0.995,
                    epsilon_min=0.01
                )
            
            # Crear entrenador
            trainer = Trainer(agent, env, max_steps_per_episode=100)
            
            # Entrenar agente
            stats = trainer.train(
                n_episodes=n_episodes,
                render_every=None,
                verbose=False
            )
            
            # Almacenar recompensas para esta ejecución
            all_rewards[algorithm].append(stats['episode_rewards'])
        
        # Calcular tiempo promedio
        execution_times[algorithm] = (time.time() - start_time) / n_runs
    
    # Programación Dinámica (solo una ejecución ya que es determinista)
    print("\nEjecutando algoritmos de Programación Dinámica...")
    
    # Crear modelo para algoritmos de programación dinámica
    model = create_model_for_dp(env)
    
    # Valor Iteración
    start_time = time.time()
    vi_agent = ValueIterationAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        model=model,
        discount_factor=0.99,
        theta=1e-6,
        max_iterations=1000
    )
    
    vi_policy, vi_value_function, vi_iterations = vi_agent.train(verbose=True)
    execution_times['Value Iteration'] = time.time() - start_time
    
    # Política Iteración
    start_time = time.time()
    pi_agent = PolicyIterationAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        model=model,
        discount_factor=0.99,
        theta=1e-6,
        max_iterations=1000
    )
    
    pi_policy, pi_value_function, pi_iterations = pi_agent.train(verbose=True)
    execution_times['Policy Iteration'] = time.time() - start_time
    
    # Visualizar curvas de aprendizaje
    plt.figure(figsize=(12, 6))
    
    for algorithm in ['Q-Learning', 'SARSA', 'Monte Carlo']:
        # Calcular promedio y desviación estándar entre ejecuciones
        mean_rewards = np.mean(all_rewards[algorithm], axis=0)
        std_rewards = np.std(all_rewards[algorithm], axis=0)
        
        # Episodios para el eje x
        episodes = np.arange(1, n_episodes + 1)
        
        # Plotear curva promedio
        plt.plot(episodes, mean_rewards, label=f'{algorithm} (Promedio)')
        
        # Sombrear área de desviación estándar
        plt.fill_between(episodes, 
                         mean_rewards - std_rewards, 
                         mean_rewards + std_rewards, 
                         alpha=0.2)
    
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa total')
    plt.title('Comparación de Algoritmos RL')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    comparison_path = os.path.join(results_dir, 'algorithm_comparison.png')
    plt.savefig(comparison_path)
    plt.close()
    
    # Imprimir tiempos de ejecución
    print("\nTiempos de ejecución promedio:")
    for algorithm, exec_time in execution_times.items():
        print(f"{algorithm}: {exec_time:.2f} segundos")
    
    # Evaluar cada agente
    print("\nEvaluando agentes entrenados...")
    
    # Crear agentes finales para evaluación
    final_agents = {
        'Q-Learning': QLearningAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            epsilon=0.0  # Sin exploración para evaluación
        ),
        'SARSA': SARSAAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            epsilon=0.0
        ),
        'Monte Carlo': MonteCarloAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            epsilon=0.0
        ),
        'Value Iteration': vi_agent,
        'Policy Iteration': pi_agent
    }
    
    # Entrenar los agentes de aprendizaje por última vez (excepto DP)
    for algorithm in ['Q-Learning', 'SARSA', 'Monte Carlo']:
        agent = final_agents[algorithm]
        trainer = Trainer(agent, env)
        trainer.train(n_episodes=n_episodes, verbose=False)
    
    # Evaluar cada agente
    eval_results = {}
    
    for algorithm, agent in final_agents.items():
        trainer = Trainer(agent, env)
        stats = trainer.evaluate(n_episodes=20, render=False)
        eval_results[algorithm] = stats
    
    # Imprimir resultados de evaluación
    print("\nResultados de evaluación:")
    print(f"{'Algoritmo':<15} {'Recompensa Media':<20} {'Longitud Media':<15}")
    print("-" * 50)
    
    for algorithm, stats in eval_results.items():
        print(f"{algorithm:<15} {stats['mean_reward']:<20.2f} {stats['mean_length']:<15.2f}")
    
    print(f"\nEjemplo completado. Se ha guardado la curva de aprendizaje en '{comparison_path}'")

if __name__ == "__main__":
    main()