It was a pre-interview test of a company. A simple comparison of two pairs of not really optimized models to see:
1. what happens if we only focus on the high loss output. Result: not worth it.
2. what happens if after maxpooling there are no activation function (ReLU). Result: Considering the probabilities of 
getting zeros in MNIST samples, not having activation functions does reasonably well, but less accurate 
than normal structure. 91.2% vs 91.6%
