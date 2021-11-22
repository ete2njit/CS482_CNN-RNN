# CS483_CNN-RNN

# Problem 1)

The fashion MNIST dataset contains 28x28 pixel pictures, which fall under one of ten categories. We can use an RNN to classify these objects by considering 28 pixels at a time, over the course of 28 timesteps. This way, each pixel will impact the final decision, while making use of RNNs 'memory' to keep track of the most meaningful features. Following the advice from Karsten Eckhardt's ['Choosing the right Hyperparameters for a simple LSTM using Keras'](https://towardsdatascience.com/choosing-the-right-hyperparameters-for-a-simple-lstm-using-keras-f8e9ed76f046), I used a single LSTM cell, as the complexity a single cell can capture should suffice for this task.

![Architecture](https://github.com/ete2njit/CS483_CNN-RNN/blob/main/resources/lstm%20architecture.png)

As stated, we use 28 pixel at each time instance, so the input size is 28. As there are 10 classes, output size is 10. Following Eckhardts advice once again, I chose the number of hidden nodes using the formula: floor(2 * input_size * output_size / 3), resulting in a hidden size of 186. After further experimentation, performance did not change much when reducing the hidden size to floor(input_size * output_size / 5) = 56, and also did not improve much when set to input_size * output_size = 280, but I chose to stay with the originally suggested 186, as speed also did not vary much between the sizes.

| Variable | Dimensions |
| -------- | ------ |
| X<sub>t</sub> | 28x1 |
| h<sub>t</sub> | 186x1 |
| s<sub>t</sub> | 186x1 |
| U | 28x186 | 
| U<sub>g</sub> | 28x186 | 
| U<sub>f</sub> | 28x186 | 
| U<sub>o</sub> | 28x186 | 
| W | 186x186 |
| W<sub>g</sub> | 186x186 |
| W<sub>f</sub> | 186x186 |
| W<sub>o</sub> | 186x186 |
| b | 186x1 |
| b<sub>g</sub> | 186x1 |
| b<sub>f</sub> | 186x1 |
| b<sub>o</sub> | 186x1 |
