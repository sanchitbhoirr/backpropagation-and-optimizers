# Short Report — Deep Learning CIA (2087101)

## Aim
Implement backpropagation from scratch for a one-hidden-layer MLP using NumPy and compare SGD with momentum against Adam using PyTorch.

## Experiment 1: Backpropagation
A sigmoid MLP was trained on a small synthetic regression dataset with MSE loss. The forward pass computes activations; the backward pass applies the chain rule to calculate gradients for weights and biases. Standard Gradient Descent updates each parameter using its gradient and learning rate.

**Observed result:** Initial MSE = 0.107561; final MSE = 0.002959 after 5,000 epochs. The MSE curve decreases substantially, showing that gradient-based updates reduced the reconstruction/prediction error on this small training dataset.

## Experiment 2: Optimizer comparison
A feedforward network was trained on a standardized synthetic two-moons classification dataset. Both optimizer runs use the same architecture, seed, train/test split, batch size and epoch count.

| Metric | SGD with momentum | Adam |
|---|---:|---:|
| Final training loss | 0.07338 | 0.07366 |
| Final test loss | 0.09040 | 0.09907 |
| Test accuracy | 96.75% | 95.75% |
| First epoch test loss below 0.1 | 7 | 5 |

In this run, Adam reached the selected test-loss threshold earlier, while SGD with momentum finished with a lower test loss and slightly higher test accuracy. Thus, faster early convergence did not guarantee the lowest final test loss in this experiment.

## Conclusion
The work demonstrates analytical backpropagation and an empirical comparison of two optimizers. Optimizer performance depends on learning rates, architecture, data and training settings; these results are specific to this experiment and do not establish a universal winner.
