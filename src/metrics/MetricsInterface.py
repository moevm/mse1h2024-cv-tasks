from abc import ABC, abstractmethod

class MetricsInterface(ABC):
    """
    Interface class for metrics.
    """
    
    def __init__(self, name):
        self.name = name
        
    @abstractmethod
    def calculate_metric(self, data):
        pass
    
    @abstractmethod
    def interpret_result(self, metric_value):
        pass
