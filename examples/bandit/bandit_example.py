#!/usr/bin/env python
# examples/bandit/bandit_example.py
"""
Ejemplo de uso del entorno Multi-Armed Bandit.
Este script demuestra diferentes estrategias para resolver el problema del bandit.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# Asegurar que el paquete principal esté en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Importar componentes de RLCore
from environments.bandit import MultiArmedBanditEnv

# Define diferentes estrategias para el problema bandit
class EpsilonGreedy:
    """Estrategia epsilon-greedy."""
    
    def __init__(self, n_arms, epsilon=0.1, step_size=0.1, optimistic_init=False, initial_value=0.0):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.step_size = step_size
        self.action_values = np.ones(n_arms) * initial_value if optimistic_init else np.zeros(n_arms)
        self.action_counts = np.zeros(n_arms, dtype=int)
        self.name = f"ε-greedy (ε={epsilon})"
    
    def select_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return np.argmax(self.action_values)
    
    def update(self, action, reward):
        self.action_counts[action] += 1
        
        # Actualización con promedio incremental
        self.action_values[action] += self.step_size * (reward - self.action_values[action])

class UCB:
    """Upper Confidence Bound strategy."""
    
    def __init__(self, n_arms, c=2.0, step_size=0.1):
        self.n_arms = n_arms
        self.c = c
        self.step_size = step_size
        self.action_values = np.zeros(n_arms)
        self.action_counts = np.zeros(n_arms, dtype=int)
        self.t = 0
        self.name = f"UCB (c={c})"
    
    def select_action(self):
        self.t += 1
        
        # Asegurar que cada brazo sea elegido al menos una vez
        for a in range(self.n_arms):
            if self.action_counts[a] == 0:
                return a
        
        # Calcular UCB para cada brazo
        ucb_values = self.action_values + self.c * np.sqrt(np.log(self.t) / self.action_counts)
        return np.argmax(ucb_values)
    
    def update(self, action, reward):
        self.action_counts[action] += 1
        
        # Actualización con promedio incremental
        self.action_values[action] += self.step_size * (reward - self.action_values[action])

class GradientBandit:
    """Estrategia de bandit basada en gradiente."""
    
    def __init__(self, n_arms, alpha=0.1, baseline=True):
        self.n_arms = n_arms
        self.alpha = alpha
        self.baseline = baseline
        self.preferences = np.zeros(n_arms)
        self.action_counts = np.zeros(n_arms, dtype=int)
        self.average_reward = 0.0
        self.t = 0
        self.name = f"Gradient (α={alpha}, baseline={baseline})"
    
    def select_action(self):
        # Softmax sobre preferencias
        exp_pref = np.exp(self.preferences)
        action_probs = exp_pref / np.sum(exp_pref)
        return np.random.choice(self.n_arms, p=action_probs)
    
    def update(self, action, reward):
        self.t += 1
        self.action_counts[action] += 1
        
        # Actualizar estimador de recompensa promedio
        if self.baseline:
            self.average_reward += (reward - self.average_reward) / self.t
        
        # Actualizar preferencias (incremento para acción seleccionada, decremento para otras)
        for a in range(self.n_arms):
            if a == action:
                # Para la acción seleccionada
                self.preferences[a] += self.alpha * (reward - self.average_reward) * (1 - self._action_probability(a))
            else:
                # Para las acciones no seleccionadas
                self.preferences[a] -= self.alpha * (reward - self.average_reward) * self._action_probability(a)
    
    def _action_probability(self, action):
        exp_pref = np.exp(self.preferences)
        return exp_pref[action] / np.sum(exp_pref)

def run_bandit_experiment(env, agent, n_steps=1000):
    """Ejecuta un experimento con un agente en un entorno bandit."""
    rewards = np.zeros(n_steps)
    optimal_actions = np.zeros(n_steps, dtype=bool)
    optimal_arm = np.argmax(env.true_values)
    regret = np.zeros(n_steps)
    
    env.reset()
    
    for t in range(n_steps):
        # Seleccionar acción
        action = agent.select_action()
        
        # Ejecutar acción
        _, reward, _, info = env.step(action)
        
        # Actualizar agente
        agent.update(action, reward)
        
        # Registrar resultados
        rewards[t] = reward
        optimal_actions[t] = (action == optimal_arm)
        regret[t] = info['regret']
    
    # Calcular resultados
    cumulative_rewards = np.cumsum(rewards)
    cumulative_regret = np.cumsum(regret)
    optimal_action_percentage = np.zeros_like(optimal_actions, dtype=float)
    
    # Calcular porcentaje de acciones óptimas a lo largo del tiempo
    window = 100  # Ventana para promedios móviles
    for t in range(n_steps):
        if t < window:
            optimal_action_percentage[t] = np.mean(optimal_actions[:t+1])
        else:
            optimal_action_percentage[t] = np.mean(optimal_actions[t-window+1:t+1])
    
    return {
        'rewards': rewards,
        'cumulative_rewards': cumulative_rewards,
        'optimal_actions': optimal_actions,
        'optimal_action_percentage': optimal_action_percentage,
        'cumulative_regret': cumulative_regret
    }

def main():
    """Función principal del ejemplo."""
    print("Ejemplo del entorno Multi-Armed Bandit")
    
    # Asegurar que la carpeta de resultados exista
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Parámetros
    n_arms = 10
    n_steps = 1000
    n_runs = 10  # Número de ejecuciones para promediar
    
    # Crear entorno
    env = MultiArmedBanditEnv(n_arms=n_arms, reward_type='gaussian')
    
    # Mostrar valores verdaderos de los brazos
    print("\nValores verdaderos de los brazos:")
    for i, value in enumerate(env.true_values):
        print(f"  Brazo {i}: {value:.4f}")
    
    # Crear agentes
    agents = [
        EpsilonGreedy(n_arms=n_arms, epsilon=0.1),
        EpsilonGreedy(n_arms=n_arms, epsilon=0.01),
        EpsilonGreedy(n_arms=n_arms, epsilon=0.1, optimistic_init=True, initial_value=5.0),
        UCB(n_arms=n_arms, c=2.0),
        GradientBandit(n_arms=n_arms, alpha=0.1)
    ]
    
    # Almacenar resultados por agente
    results = {agent.name: {'cumulative_rewards': [], 'optimal_action_percentage': [], 'cumulative_regret': []} 
               for agent in agents}
    
    # Ejecutar experimentos
    for run in range(n_runs):
        print(f"\nEjecutando experimento {run+1}/{n_runs}...")
        
        # Crear nuevo entorno para cada ejecución
        env = MultiArmedBanditEnv(n_arms=n_arms, reward_type='gaussian')
        
        # Ejecutar cada agente
        for agent in agents:
            # Reiniciar agente
            if hasattr(agent, 'reset'):
                agent.reset()
            else:
                # Recrear agente del mismo tipo
                if isinstance(agent, EpsilonGreedy):
                    agent = EpsilonGreedy(n_arms=n_arms, epsilon=agent.epsilon, 
                                         optimistic_init=agent.action_values[0] > 0)
                elif isinstance(agent, UCB):
                    agent = UCB(n_arms=n_arms, c=agent.c)
                elif isinstance(agent, GradientBandit):
                    agent = GradientBandit(n_arms=n_arms, alpha=agent.alpha, baseline=agent.baseline)
            
            # Ejecutar experimento
            run_results = run_bandit_experiment(env, agent, n_steps=n_steps)
            
            # Almacenar resultados
            results[agent.name]['cumulative_rewards'].append(run_results['cumulative_rewards'])
            results[agent.name]['optimal_action_percentage'].append(run_results['optimal_action_percentage'])
            results[agent.name]['cumulative_regret'].append(run_results['cumulative_regret'])
    
    # Calcular promedios entre ejecuciones
    for agent_name in results:
        for metric in results[agent_name]:
            results[agent_name][metric] = np.mean(results[agent_name][metric], axis=0)
    
    # Visualizar resultados
    plt.figure(figsize=(18, 5))
    
    # Graficar porcentaje de acciones óptimas
    plt.subplot(1, 3, 1)
    for agent_name, agent_results in results.items():
        plt.plot(agent_results['optimal_action_percentage'], label=agent_name)
    plt.xlabel('Pasos')
    plt.ylabel('% Acciones Óptimas')
    plt.title('Porcentaje de Acciones Óptimas')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Graficar recompensa acumulada
    plt.subplot(1, 3, 2)
    for agent_name, agent_results in results.items():
        plt.plot(agent_results['cumulative_rewards'], label=agent_name)
    plt.xlabel('Pasos')
    plt.ylabel('Recompensa Acumulada')
    plt.title('Recompensa Acumulada')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Graficar regret acumulado
    plt.subplot(1, 3, 3)
    for agent_name, agent_results in results.items():
        plt.plot(agent_results['cumulative_regret'], label=agent_name)
    plt.xlabel('Pasos')
    plt.ylabel('Regret Acumulado')
    plt.title('Regret Acumulado')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar la figura en la carpeta de resultados
    result_path = os.path.join(results_dir, 'bandit_comparison.png')
    plt.savefig(result_path)
    plt.close()
    
    print("\nResultados finales:")
    for agent_name, agent_results in results.items():
        print(f"{agent_name}:")
        print(f"  Recompensa acumulada final: {agent_results['cumulative_rewards'][-1]:.2f}")
        print(f"  Porcentaje final de acciones óptimas: {agent_results['optimal_action_percentage'][-1]*100:.2f}%")
        print(f"  Regret acumulado final: {agent_results['cumulative_regret'][-1]:.2f}")
    
    print(f"\nEjemplo completado. Se ha guardado la comparación en '{result_path}'")

if __name__ == "__main__":
    main()