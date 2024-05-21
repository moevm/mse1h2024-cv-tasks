from abc import ABC, abstractmethod 

# Creating an abstract base class named MetricsInterface inheriting from ABC
class MetricsInterface(ABC):
    """
    Interface class for metrics.
    """
    
    # Constructor method that initializes the object with a name
    def __init__(self, name):
        self.name = name  # Initializing the name attribute
    
    # Abstract method for calculating the metric
    @abstractmethod
    def calculate_metric(self, data):
        pass  # Placeholder for the method to be implemented by subclasses
    
    # Abstract method for interpreting the result of the metric calculation
    @abstractmethod
    def interpret_result(self, metric_value):
        pass  # Placeholder for the method to be implemented by subclasses
