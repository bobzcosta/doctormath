from abc import ABC, abstractmethod
import numpy as np

class BaseSolver(ABC):
    """كل محرك حل يجب أن يرث من هذا الصنف."""
    
    @abstractmethod
    def solve(self, A, b, **kwargs):
        """حل النظام A x = b. يجب أن يعيد numpy array."""
        pass
    
    @property
    def name(self):
        return self.__class__.__name__
