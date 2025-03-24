# rlcore/utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_learning_curve(rewards, window=10, title="Learning Curve", figsize=(10, 6)):
    """
    Visualiza la curva de aprendizaje (recompensas por episodio).
    
    Args:
        rewards: Lista de recompensas por episodio
        window: Tamaño de la ventana para suavizado
        title: Título del gráfico
        figsize: Tamaño de la figura
    """
    plt.figure(figsize=figsize)
    
    # Datos originales
    episodes = np.arange(len(rewards))
    plt.plot(episodes, rewards, alpha=0.3, label='Original')
    
    # Promedio móvil para suavizado
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
        plt.plot(episodes[(window-1)//2:-(window-1)//2], smoothed, 
                 label=f'Suavizado (ventana={window})')
    
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa total')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt


def plot_value_function(value_function, shape, title="Función de Valor"):
    """
    Visualiza una función de valor para un entorno de rejilla 2D.
    
    Args:
        value_function: Array 1D con valores para cada estado
        shape: Tupla (height, width) de la rejilla
        title: Título del gráfico
    """
    fig = plt.figure(figsize=(10, 8))
    
    # Reformatear función de valor a 2D
    value_grid = value_function.reshape(shape)
    
    # Crear mapa de calor
    ax = sns.heatmap(value_grid, cmap='viridis', annot=True, fmt='.2f', cbar=True)
    ax.invert_yaxis()  # Para que (0,0) esté en la esquina superior izquierda
    
    plt.title(title)
    plt.tight_layout()
    
    return plt


def plot_policy_arrows(policy, shape, title="Política"):
    """
    Visualiza una política para un entorno de rejilla 2D usando flechas.
    Asume acciones: 0=arriba, 1=derecha, 2=abajo, 3=izquierda
    
    Args:
        policy: Array 1D con acciones para cada estado
        shape: Tupla (height, width) de la rejilla
        title: Título del gráfico
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Reformatear política a 2D
    policy_grid = policy.reshape(shape)
    
    # Matrices para componentes X e Y de las flechas
    X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    
    # Convertir acciones a direcciones
    U = np.zeros_like(policy_grid, dtype=float)  # Componente X
    V = np.zeros_like(policy_grid, dtype=float)  # Componente Y
    
    # Mapear acciones a direcciones
    action_to_dir = {
        0: (0, -1),   # Arriba: (0, -1)
        1: (1, 0),    # Derecha: (1, 0)
        2: (0, 1),    # Abajo: (0, 1)
        3: (-1, 0)    # Izquierda: (-1, 0)
    }
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            action = policy_grid[i, j]
            if action in action_to_dir:
                dx, dy = action_to_dir[action]
                U[i, j] = dx
                V[i, j] = dy
    
    # Dibujar flechas
    ax.quiver(X, Y, U, V, scale=20, units='xy', width=0.05)
    
    # Configuración del gráfico
    ax.set_xlim(-0.5, shape[1] - 0.5)
    ax.set_ylim(-0.5, shape[0] - 0.5)
    ax.set_xticks(np.arange(shape[1]))
    ax.set_yticks(np.arange(shape[0]))
    ax.grid(True)
    ax.invert_yaxis()  # Para que (0,0) esté en la esquina superior izquierda
    
    plt.title(title)
    plt.tight_layout()
    
    return plt


def plot_action_values(q_values, state, action_names=None, title="Valores Q"):
    """
    Visualiza los valores Q para un estado específico.
    
    Args:
        q_values: Matriz de valores Q o diccionario {(estado, acción): valor}
        state: Estado para el cual visualizar valores Q
        action_names: Lista de nombres de acciones
        title: Título del gráfico
    """
    # Extraer valores Q para el estado dado
    if isinstance(q_values, dict):
        actions = [a for s, a in q_values.keys() if s == state]
        values = [q_values[(state, a)] for a in actions]
    else:
        values = q_values[state]
        actions = np.arange(len(values))
    
    # Usar nombres de acciones si se proporcionan
    if action_names is not None:
        x_labels = action_names
    else:
        x_labels = [f"Acción {a}" for a in actions]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(x_labels, values, color='skyblue')
    
    # Resaltar la mejor acción
    best_idx = np.argmax(values)
    bars[best_idx].set_color('tomato')
    
    plt.xlabel('Acción')
    plt.ylabel('Valor Q')
    plt.title(f"{title} para Estado {state}")
    plt.grid(True, alpha=0.3)
    
    return plt