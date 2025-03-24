# rlcore/core/experience.py
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class Experience:
    """
    Representa una única experiencia (transición) en RL.
    
    Attributes:
        state: Estado antes de la acción
        action: Acción tomada
        reward: Recompensa recibida
        next_state: Estado resultante
        done: Indicador de fin de episodio
        info: Información adicional opcional
    """
    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool
    info: Optional[dict] = None


@dataclass
class Episode:
    """
    Representa un episodio completo como una secuencia de experiencias.
    
    Attributes:
        experiences: Lista de experiencias
        total_reward: Recompensa total acumulada
    """
    experiences: list
    
    @property
    def total_reward(self):
        """Calcula la recompensa total del episodio."""
        return sum(exp.reward for exp in self.experiences)
    
    @property
    def length(self):
        """Devuelve la longitud del episodio."""
        return len(self.experiences)