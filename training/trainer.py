# rlcore/training/trainer.py
import time
import numpy as np
from core.experience import Episode
from utils.visualization import plot_learning_curve

class Trainer:
    """
    Clase para entrenar agentes RL en entornos.
    Maneja la interacción entre agente y entorno, recopila estadísticas
    y proporciona visualizaciones.
    """
    
    def __init__(self, agent, env, max_steps_per_episode=1000):
        """
        Inicializa el entrenador.
        
        Args:
            agent: Agente de RL a entrenar
            env: Entorno donde el agente actuará
            max_steps_per_episode: Número máximo de pasos por episodio
        """
        self.agent = agent
        self.env = env
        self.max_steps_per_episode = max_steps_per_episode
        
        # Estadísticas
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_time = 0
    
    def train(self, n_episodes, render_every=None, verbose=True, render_mode='human'):
        """
        Entrena al agente durante un número determinado de episodios.
        
        Args:
            n_episodes: Número de episodios de entrenamiento
            render_every: Renderizar cada N episodios (None = no renderizar)
            verbose: Si True, imprime estadísticas durante el entrenamiento
            render_mode: Modo de renderizado ('human', 'rgb_array', etc.)
            
        Returns:
            dict: Estadísticas de entrenamiento
        """
        start_time = time.time()
        
        for episode in range(1, n_episodes + 1):
            # Reiniciar entorno y agente
            state = self.env.reset()
            self.agent.reset()
            
            episode_reward = 0
            episode_experiences = []
            
            # Renderizar si es necesario
            if render_every and episode % render_every == 0:
                self.env.render(mode=render_mode)
            
            # Bucle de episodio
            for step in range(1, self.max_steps_per_episode + 1):
                # Seleccionar y ejecutar acción
                action = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                # Acumular recompensa
                episode_reward += reward
                
                # Almacenar experiencia
                experience = (state, action, reward, next_state, done)
                episode_experiences.append(experience)
                
                # Actualizar agente
                self.agent.update(experience)
                
                # Renderizar si es necesario
                if render_every and episode % render_every == 0:
                    self.env.render(mode=render_mode)
                
                # Actualizar estado
                state = next_state
                
                if done:
                    break
            
            # Crear objeto Episode y almacenarlo si se necesita
            episode_obj = Episode(episode_experiences)
            
            # Actualizar estadísticas
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step)
            
            # Mostrar progreso
            if verbose and episode % (n_episodes // 10 or 1) == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])  # Promedio de últimos 100 episodios
                print(f"Episodio {episode}/{n_episodes} - "
                      f"Recompensa: {episode_reward:.2f} - "
                      f"Promedio último 100: {avg_reward:.2f} - "
                      f"Pasos: {step}")
        
        self.training_time = time.time() - start_time
        
        # Crear resumen de estadísticas
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_time': self.training_time,
            'n_episodes': n_episodes,
            'final_avg_reward': np.mean(self.episode_rewards[-100:])
        }
        
        if verbose:
            print(f"\nEntrenamiento completado en {self.training_time:.2f} segundos")
            print(f"Recompensa promedio final (100 ep.): {stats['final_avg_reward']:.2f}")
        
        return stats
    
    def plot_rewards(self, window=10, title=None):
        """
        Visualiza la curva de recompensas del entrenamiento.
        
        Args:
            window: Tamaño de la ventana para suavizado
            title: Título del gráfico
            
        Returns:
            matplotlib.pyplot: Objeto plot para personalización adicional
        """
        if not title:
            title = "Curva de Aprendizaje"
        return plot_learning_curve(self.episode_rewards, window, title)
    
    def evaluate(self, n_episodes=10, render=False, render_mode='human'):
        """
        Evalúa el agente entrenado sin exploración.
        
        Args:
            n_episodes: Número de episodios de evaluación
            render: Si True, renderiza el entorno
            render_mode: Modo de renderizado
            
        Returns:
            dict: Estadísticas de evaluación
        """
        # Guardar epsilon original y establecerlo a 0 para evaluación (sin exploración)
        if hasattr(self.agent, 'policy') and hasattr(self.agent.policy, 'epsilon'):
            original_epsilon = self.agent.policy.epsilon
            self.agent.policy.epsilon = 0
        
        eval_rewards = []
        eval_lengths = []
        
        for episode in range(1, n_episodes + 1):
            state = self.env.reset()
            self.agent.reset()
            
            episode_reward = 0
            steps = 0
            
            for step in range(1, self.max_steps_per_episode + 1):
                if render:
                    self.env.render(mode=render_mode)
                
                action = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(steps)
        
        # Restaurar epsilon original
        if hasattr(self.agent, 'policy') and hasattr(self.agent.policy, 'epsilon'):
            self.agent.policy.epsilon = original_epsilon
        
        # Estadísticas de evaluación
        stats = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'min_reward': np.min(eval_rewards),
            'max_reward': np.max(eval_rewards),
            'mean_length': np.mean(eval_lengths)
        }
        
        return stats