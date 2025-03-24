#!/usr/bin/env python
# examples/gridworld/gridworld_q_learning.py
"""
Ejemplo de uso de Q-Learning en un entorno GridWorld simple.
Este script muestra cómo utilizar los componentes básicos de RLCore
para entrenar un agente Q-Learning en un entorno de cuadrícula.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Asegurar que el paquete principal esté en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Importar componentes de RLCore
from agents.q_learning import QLearningAgent
from environments.gridworld import GridWorldEnv
from training.trainer import Trainer
from utils.visualization import plot_value_function, plot_policy_arrows

def main():
    """Función principal del ejemplo."""
    print("Ejemplo de Q-Learning en GridWorld")
    
    # Asegurar que la carpeta de resultados exista
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Crear un entorno GridWorld con obstáculos
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
    
    # Crear agente Q-Learning
    agent = QLearningAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,  # Comenzar con alta exploración
        epsilon_decay=0.995,  # Decaimiento gradual
        epsilon_min=0.01  # Exploración mínima residual
    )
    
    # Crear entrenador
    trainer = Trainer(agent, env, max_steps_per_episode=100)
    
    # Entrenar agente
    print("\nIniciando entrenamiento...")
    stats = trainer.train(
        n_episodes=500,
        render_every=None,  # Comentar esta línea para visualizar cada N episodios
        verbose=True
    )
    
    # Visualizar curva de aprendizaje
    plt.figure(figsize=(10, 6))
    ax = trainer.plot_rewards(window=20, title="Curva de Aprendizaje: Q-Learning en GridWorld")
    rewards_path = os.path.join(results_dir, 'qlearning_rewards.png')
    plt.savefig(rewards_path)
    plt.close()
    
    # Extraer y visualizar la política y función de valor
    # Convertir la tabla Q en una función de valor y política
    q_table = agent.q_table
    width, height = env.width, env.height
    value_function = np.zeros(width * height)
    policy = np.zeros(width * height, dtype=int)
    
    # Para cada estado, obtener el valor máximo y la acción correspondiente
    for state in range(width * height):
        q_values = [q_table.get((state, a), 0.0) for a in range(4)]
        
        # Verificar si el estado es un obstáculo (opcional)
        pos = env._get_position(state)
        if pos in obstacles:
            # Para obstáculos, podemos asignar valores específicos o dejarlos en 0
            value_function[state] = 0
            policy[state] = 0
            continue
            
        # Para estados normales, obtener valor máximo y mejor acción
        max_q = max(q_values)
        best_action = np.argmax(q_values)
        value_function[state] = max_q
        policy[state] = best_action
    
    # Visualizar función de valor
    plt.figure(figsize=(8, 6))
    plot_value_function(value_function, (height, width), title="Función de Valor Q")
    value_path = os.path.join(results_dir, 'qlearning_value_function.png')
    plt.savefig(value_path)
    plt.close()
    
    # Visualizar política
    plt.figure(figsize=(8, 6))
    plot_policy_arrows(policy, (height, width), title="Política Q-Learning")
    policy_path = os.path.join(results_dir, 'qlearning_policy.png')
    plt.savefig(policy_path)
    plt.close()
    
    # Evaluar el agente entrenado
    print("\nEvaluando agente entrenado...")
    eval_stats = trainer.evaluate(n_episodes=10, render=True)
    print(f"Recompensa media: {eval_stats['mean_reward']:.2f}")
    print(f"Longitud media de episodio: {eval_stats['mean_length']:.2f}")
    
    # Guardar el modelo
    model_path = os.path.join(results_dir, 'qlearning_agent.pkl')
    agent.save(model_path)
    
    print("\nEjemplo completado. Se han guardado los siguientes archivos:")
    print(f"  - Curva de aprendizaje: '{rewards_path}'")
    print(f"  - Función de valor: '{value_path}'")
    print(f"  - Política: '{policy_path}'")
    print(f"  - Modelo del agente: '{model_path}'")

if __name__ == "__main__":
    main()