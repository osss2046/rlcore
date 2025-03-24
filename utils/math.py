# rlcore/utils/math.py
import numpy as np

def discount_rewards(rewards, gamma):
    """
    Calcula los retornos descontados (G_t) a partir de recompensas.
    Fórmula: G_t = R_{t+1} + gamma*R_{t+2} + gamma^2*R_{t+3} + ...
    
    Args:
        rewards: Lista de recompensas [r_1, r_2, ..., r_T]
        gamma: Factor de descuento (0 <= gamma <= 1)
        
    Returns:
        np.array: Retornos descontados para cada paso de tiempo
    """
    discounted = np.zeros_like(rewards, dtype=float)
    running_sum = 0
    
    for t in reversed(range(len(rewards))):
        running_sum = rewards[t] + gamma * running_sum
        discounted[t] = running_sum
        
    return discounted


def compute_advantages(rewards, values, gamma=0.99, lambda_=0.95):
    """
    Calcula ventajas generalizadas (GAE) según Schulman et al.
    
    Args:
        rewards: Lista de recompensas
        values: Estimaciones de valor para cada estado
        gamma: Factor de descuento
        lambda_: Factor de suavizado GAE
        
    Returns:
        np.array: Ventajas estimadas
    """
    deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
    advantages = np.zeros_like(deltas, dtype=float)
    
    advantage = 0
    for t in reversed(range(len(deltas))):
        advantage = deltas[t] + gamma * lambda_ * advantage
        advantages[t] = advantage
        
    return advantages


def compute_td_error(state_value, next_state_value, reward, gamma, done):
    """
    Calcula el error TD para actualización de función de valor.
    
    Args:
        state_value: Valor estimado del estado actual
        next_state_value: Valor estimado del estado siguiente
        reward: Recompensa recibida
        gamma: Factor de descuento
        done: Indicador de fin de episodio
        
    Returns:
        float: Error TD
    """
    if done:
        return reward - state_value
    else:
        return reward + gamma * next_state_value - state_value


def normalize(x, epsilon=1e-8):
    """
    Normaliza un array para que tenga media 0 y desviación estándar 1.
    
    Args:
        x: Array a normalizar
        epsilon: Pequeña constante para evitar división por cero
        
    Returns:
        np.array: Array normalizado
    """
    return (x - x.mean()) / (x.std() + epsilon)