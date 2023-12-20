# cmo_digits

Handwritten Digit recogniser done as part of the Computational Methods and Optimisation Course at Plaksha

## Training

You can import the `Network` class and train it with different hyperparameters and training data or
you can use `train.py`.

Check the usage of `train.py`:

- `python3 train.py --help`

## Results

### SGD

Network Hyperparameters and Properties

- number of epochs: 30
- mini batch size: 10
- learning rate: 3.0
- activation function: sigmoid

Results:

Predicted 9537 out of 10000 numbers correctly.

## Resources & References

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). _Deep Learning_. MIT Press.
- Nielsen, M. (2015). _Neural Networks and Deep Learning_. Determination Press. https://neuralnetworksanddeeplearning.com/
- Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
- Hasanpour, S. H., Rouhani, M., Fayyaz, M., & Sabokrou, M. (2016). Lets keep it simple, using simple architectures to outperform deeper and more complex architectures. arXiv preprint arXiv:1608.06037.
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision (pp. 1026-1034).
