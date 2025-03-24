#!/usr/bin/env python
# examples/deeprl/deeprl_cartpole.py
"""
Ejemplo de uso de algoritmos Deep RL en el entorno CartPole.
Este script compara diferentes algoritmos de Deep RL:
- DQN (Deep Q-Network)
- REINFORCE (Policy Gradient)
- A2C (Advantage Actor-Critic)
- PPO (Proximal Policy Optimization)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import torch

# Asegurar que el paquete principal esté en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Importar componentes de RLCore
from environments.cartpole import CartPoleEnv
from agents.dqn_pytorch import DQNAgent, DuelingDQNAgent
from agents.policy_gradient import REINFORCEAgent
from agents.actor_critic import A2CAgent, PPOAgent
from training.trainer import Trainer

def create_agent(agent_type, env, hidden_dims=[64, 64]):
    """
    Crea un agente según el tipo especificado.
    
    Args:
        agent_type: Tipo de agente ('dqn', 'dueling_dqn', 'reinforce', 'a2c', 'ppo')
        env: Entorno en el que se usará el agente
        hidden_dims: Dimensiones de las capas ocultas
        
    Returns:
        Agent: Instancia del agente especificado
    """
    if agent_type == 'dqn':
        return DQNAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            learning_rate=0.001,
            discount_factor=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=10000,
            batch_size=64,
            target_update_freq=10,
            hidden_dims=hidden_dims
        )
    elif agent_type == 'dueling_dqn':
        return DuelingDQNAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            learning_rate=0.001,
            discount_factor=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=10000,
            batch_size=64,
            target_update_freq=10,
            hidden_dims=hidden_dims,
            use_double_dqn=True
        )
    elif agent_type == 'reinforce':
        return REINFORCEAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            learning_rate=0.001,
            discount_factor=0.99,
            hidden_dims=hidden_dims,
            use_baseline=True
        )
    elif agent_type == 'a2c':
        return A2CAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            learning_rate=0.001,
            discount_factor=0.99,
            hidden_dims=hidden_dims,
            shared_layers=True,
            entropy_coef=0.01,
            value_coef=0.5
        )
    elif agent_type == 'ppo':
        return PPOAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            learning_rate=0.0003,
            discount_factor=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            hidden_dims=hidden_dims,
            shared_layers=True,
            entropy_coef=0.01,
            value_coef=0.5,
            epochs=4,
            batch_size=64
        )
    else:
        raise ValueError(f"Tipo de agente no soportado: {agent_type}")

def train_agent(agent_type, n_episodes=500, render_every=None, seed=None):
    """
    Entrena un agente en el entorno CartPole.
    
    Args:
        agent_type: Tipo de agente ('dqn', 'dueling_dqn', 'reinforce', 'a2c', 'ppo')
        n_episodes: Número de episodios de entrenamiento
        render_every: Renderizar cada N episodios (None = no renderizar)
        seed: Semilla para reproducibilidad
        
    Returns:
        tuple: (recompensas, tiempo_entrenamiento, agente)
    """
    # Asegurar directorio de resultados
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Crear entorno y establecer semilla si se proporciona
    env = CartPoleEnv(max_episode_steps=500)
    if seed is not None:
        env.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Crear agente
    agent = create_agent(agent_type, env)
    
    # Crear entrenador
    trainer = Trainer(agent, env, max_steps_per_episode=500)
    
    # Medir tiempo de entrenamiento
    start_time = time.time()
    
    # Entrenar agente
    print(f"\nEntrenando agente {agent_type.upper()}...")
    stats = trainer.train(
        n_episodes=n_episodes,
        render_every=render_every,
        verbose=True
    )
    
    # Calcular tiempo de entrenamiento
    training_time = time.time() - start_time
    print(f"Tiempo de entrenamiento: {training_time:.2f} segundos")
    
    # Guardar modelo
    model_path = os.path.join(results_dir, f"{agent_type}_cartpole.pt")
    agent.save(model_path)
    print(f"Modelo guardado en: {model_path}")
    
    # Guardar curva de aprendizaje
    plt.figure(figsize=(10, 6))
    trainer.plot_rewards(window=20, title=f"Curva de Aprendizaje: {agent_type.upper()} en CartPole")
    curve_path = os.path.join(results_dir, f"{agent_type}_learning_curve.png")
    plt.savefig(curve_path)
    plt.close()
    
    # Evaluar agente
    print("\nEvaluando agente entrenado...")
    eval_stats = trainer.evaluate(n_episodes=10, render=False)
    print(f"Recompensa media: {eval_stats['mean_reward']:.2f}")
    print(f"Longitud media de episodio: {eval_stats['mean_length']:.2f}")
    
    # Mostrar un episodio con renderizado
    print("\nMostrando un episodio con el agente entrenado:")
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        env.render()
        action = agent.select_action(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
    
    print(f"Recompensa total del episodio: {total_reward}")
    
    return stats['episode_rewards'], training_time, agent

def compare_agents(agent_types, n_episodes=500, n_runs=3, seed=None):
    """
    Compara diferentes tipos de agentes en el entorno CartPole.
    
    Args:
        agent_types: Lista de tipos de agentes a comparar
        n_episodes: Número de episodios de entrenamiento por agente
        n_runs: Número de ejecuciones para promediar resultados
        seed: Semilla inicial para reproducibilidad
    """
    # Asegurar directorio de resultados
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Almacenar resultados por agente
    results = {agent_type: {'rewards': [], 'times': []} for agent_type in agent_types}
    
    # Entrenar cada agente
    for agent_type in agent_types:
        print(f"\n{'='*50}")
        print(f"Entrenando agente {agent_type.upper()}")
        print(f"{'='*50}")
        
        for run in range(n_runs):
            print(f"\nEjecución {run+1}/{n_runs}")
            
            # Usar semilla diferente para cada ejecución pero determinista
            run_seed = seed + run if seed is not None else None
            
            # Entrenar agente
            rewards, train_time, _ = train_agent(agent_type, n_episodes, render_every=None, seed=run_seed)
            
            # Almacenar resultados
            results[agent_type]['rewards'].append(rewards)
            results[agent_type]['times'].append(train_time)
    
    # Calcular promedios
    for agent_type in agent_types:
        results[agent_type]['mean_rewards'] = np.mean(results[agent_type]['rewards'], axis=0)
        results[agent_type]['std_rewards'] = np.std(results[agent_type]['rewards'], axis=0)
        results[agent_type]['mean_time'] = np.mean(results[agent_type]['times'])
    
    # Visualizar comparación
    plt.figure(figsize=(12, 8))
    
    # Graficar curvas de aprendizaje
    for agent_type in agent_types:
        mean_rewards = results[agent_type]['mean_rewards']
        std_rewards = results[agent_type]['std_rewards']
        episodes = np.arange(1, n_episodes + 1)
        
        plt.plot(episodes, mean_rewards, label=f"{agent_type.upper()}")
        plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
    
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa total')
    plt.title('Comparación de Algoritmos Deep RL en CartPole')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Guardar gráfica
    comparison_path = os.path.join(results_dir, "deeprl_comparison.png")
    plt.savefig(comparison_path)
    plt.close()
    
    # Imprimir resumen de tiempos
    print("\nResumen de tiempos de entrenamiento:")
    for agent_type in agent_types:
        print(f"{agent_type.upper()}: {results[agent_type]['mean_time']:.2f} segundos")
    
    print(f"\nComparación guardada en: {comparison_path}")

def main():
    """Función principal del ejemplo."""
    parser = argparse.ArgumentParser(description='Ejemplo de Deep RL en CartPole')
    parser.add_argument('--agent', type=str, default='dqn',
                        choices=['dqn', 'dueling_dqn', 'reinforce', 'a2c', 'ppo'],
                        help='Tipo de agente a entrenar')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Número de episodios de entrenamiento')
    parser.add_argument('--compare', action='store_true',
                        help='Comparar todos los algoritmos')
    parser.add_argument('--render', action='store_true',
                        help='Renderizar durante el entrenamiento')
    parser.add_argument('--seed', type=int, default=None,
                        help='Semilla para reproducibilidad')
    
    args = parser.parse_args()
    
    if args.compare:
        # Comparar todos los algoritmos
        compare_agents(
            agent_types=['dqn', 'dueling_dqn', 'reinforce', 'a2c', 'ppo'],
            n_episodes=args.episodes,
            n_runs=3,
            seed=args.seed
        )
    else:
        # Entrenar un solo agente
        render_every = 50 if args.render else None
        train_agent(
            agent_type=args.agent,
            n_episodes=args.episodes,
            render_every=render_every,
            seed=args.seed
        )

if __name__ == "__main__":
    main()