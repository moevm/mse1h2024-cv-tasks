import torch
from torch.utils.data import DataLoader
from SimpleCNN import SimpleCNN
from SimpleDataset import SimpleDataset
from TestMetrics import calculate_metrics

class ModelEvaluator:
    def __init__(self, model, dataset, batch_size):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size

    def evaluate(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        predictions = []
        ground_truth = []

        for batch in dataloader:
            inputs, targets = batch
            outputs = self.model(inputs)
            predictions.extend(outputs.argmax(dim=1).tolist())
            ground_truth.extend(targets.tolist())

        metrics = calculate_metrics(predictions, ground_truth)

        return metrics

# Example usage
if __name__ == "__main__":
    data = ... 
    labels = ...
    model = SimpleCNN()  # Instantiate the SimpleCNN model
    dataset = SimpleDataset(data, labels)  # Instantiate the SimpleDataset dataset
    batch_size = 64

    evaluator = ModelEvaluator(model, dataset, batch_size)
    metrics = evaluator.evaluate()
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")

