# CS483_CNN-RNN

# Problem 1, Architecture)

The fashion MNIST dataset contains 28x28 pixel pictures, which each fall under one of ten categories. We can use an RNN to classify these objects by considering 28 pixels at a time, over the course of 28 timesteps. This way, each pixel will impact the final decision, while making use of RNNs 'memory' to keep track of the most meaningful features. Following the advice from Karsten Eckhardt's ['Choosing the right Hyperparameters for a simple LSTM using Keras'](https://towardsdatascience.com/choosing-the-right-hyperparameters-for-a-simple-lstm-using-keras-f8e9ed76f046), I used a single LSTM cell, as the complexity a single cell can capture should suffice for this task.

![Architecture](https://github.com/ete2njit/CS483_CNN-RNN/blob/main/resources/lstm%20architecture.png)

As stated, we use 28 pixel at each time instance, so the input size is 28. As there are 10 classes, output size is 10. Following Eckhardts advice once again, I chose the number of hidden nodes using the formula: floor(2 * input_size * output_size / 3), resulting in a hidden size of 186. After further experimentation, performance, both accuracy and speed, did not change much when reducing the hidden size to floor(input_size * output_size / 5) = 56, or when increasing it to input_size * output_size = 280, so I chose to stay with the originally suggested 186.

| Variable(s) | Dimensions |
| -------- | ------ |
| X<sub>t</sub> | 28x1 |
| h<sub>t</sub> | 186x1 |
| s<sub>t</sub> | 186x1 |
| W, W<sub>g</sub>, W<sub>f</sub>, W<sub>o</sub> | 186x186 |
| U, U<sub>g</sub>, U<sub>f</sub>, U<sub>o</sub>  | 28x186 | 
| b, b<sub>g</sub>, b<sub>f</sub>, b<sub>o</sub> | 186x1 |

# Problem 2, Implementation)

[Here](https://colab.research.google.com/drive/1tf-U-YScYBok_h41_JqFSmZFpwBjQk5z?usp=sharing) is the implementation of the LSTM architecture laid out above, as well as a CNN implementation.

# Problem 3, Performance

LSTM performance:
![](https://github.com/ete2njit/CS482_CNN-RNN/blob/main/resources/LSTM_performance.png)

CNN performance:
![](https://github.com/ete2njit/CS482_CNN-RNN/blob/main/resources/CNN_performance.png)

As can be seen, both models perform similarly well, with the LSTM model training slightly faster, at 13 miliseconds per step compared to the CNN-models 16 miliseconds, while the CNN model pulls slightly ahead in accuracy, at a peak and final validation accuracy of 91.3 and 90.94, respectively, whereas the single layer LSTM reaches a peak accuracy of 90.24 and a final accuracy of 89.96 on the validation set. In an effort to bring the accuracies closer together, I tried using a model with two stacked LSTM layers, which resulted in the following performance:

2 layer LSTM performance:
![](https://github.com/ete2njit/CS482_CNN-RNN/blob/main/resources/LSTM_2_layer_performance.png)
