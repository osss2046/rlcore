# rlcore/core/space.py
import numpy as np

class Space:
    """
    Clase base para espacios de acciones y observaciones.
    """
    
    def contains(self, x):
        """
        Verifica si x es un elemento válido del espacio.
        
        Args:
            x: El elemento a verificar
            
        Returns:
            bool: True si x pertenece al espacio, False en caso contrario
        """
        raise NotImplementedError
    
    def sample(self):
        """
        Muestrea un elemento aleatorio del espacio.
        
        Returns:
            Un elemento válido del espacio
        """
        raise NotImplementedError


class Discrete(Space):
    """
    Espacio discreto con n elementos {0, 1, ..., n-1}.
    """
    
    def __init__(self, n):
        """
        Inicializa un espacio discreto.
        
        Args:
            n: Número de elementos en el espacio
        """
        self.n = n
    
    def sample(self):
        """
        Muestrea un elemento aleatorio.
        
        Returns:
            int: Un entero entre 0 y n-1
        """
        return np.random.randint(0, self.n)
    
    def contains(self, x):
        """
        Verifica si x es un elemento válido.
        
        Args:
            x: El elemento a verificar
            
        Returns:
            bool: True si x es un entero entre 0 y n-1
        """
        if isinstance(x, int):
            return 0 <= x < self.n
        return False


class Box(Space):
    """
    Espacio continuo de n dimensiones con límites por dimensión.
    """
    
    def __init__(self, low, high, shape=None, dtype=np.float32):
        """
        Inicializa un espacio Box.
        
        Args:
            low: Límite inferior (escalar o array)
            high: Límite superior (escalar o array)
            shape: Forma del espacio
            dtype: Tipo de datos
        """
        if shape is None:
            if np.isscalar(low) and np.isscalar(high):
                raise ValueError("Shape must be provided when low and high are scalars")
            low = np.asarray(low, dtype=dtype)
            high = np.asarray(high, dtype=dtype)
            shape = low.shape
        
        self.low = np.full(shape, low if np.isscalar(low) else low.flat[0], dtype=dtype)
        self.high = np.full(shape, high if np.isscalar(high) else high.flat[0], dtype=dtype)
        self.shape = shape
        self.dtype = dtype
    
    def sample(self):
        """
        Muestrea un punto aleatorio dentro de los límites.
        
        Returns:
            np.ndarray: Un punto aleatorio en el espacio
        """
        return np.random.uniform(
            low=self.low, 
            high=self.high, 
            size=self.shape
        ).astype(self.dtype)
    
    def contains(self, x):
        """
        Verifica si x está dentro de los límites.
        
        Args:
            x: El punto a verificar
            
        Returns:
            bool: True si x está dentro de los límites
        """
        x = np.asarray(x)
        return (
            x.shape == self.shape and 
            np.all(x >= self.low) and 
            np.all(x <= self.high)
        )