A simple comparison of two pairs of models to see:
1. what happens if we only focus on the high loss output. Result: not worth it.
2. what happens if after maxpooling there are no activation function (ReLU). Result: Considering the probabilities of 
getting zeros in MNIST samples, not having activation functions does reasonably well, but less accurate 
than normal structure.
