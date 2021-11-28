# Problem 1, Architecture)

The fashion MNIST dataset contains 28x28 pixel pictures, which each fall under one of ten categories. We can use an RNN to classify these objects by considering 28 pixels at a time, over the course of 28 timesteps. This way, each pixel will impact the final decision, while making use of RNNs 'memory' to keep track of the most meaningful features. Following the advice from Karsten Eckhardt's ['Choosing the right Hyperparameters for a simple LSTM using Keras'](https://towardsdatascience.com/choosing-the-right-hyperparameters-for-a-simple-lstm-using-keras-f8e9ed76f046), I used a single LSTM cell, as the complexity a single cell can capture should suffice for this task.

![Architecture](https://github.com/ete2njit/CS483_CNN-RNN/blob/main/resources/lstm_architecture.png)

The final output of this cell, h<sub>28</sub>, is then fed into a fully connected network with 10 outputs, one for each class.

As stated, we use 28 pixel at each time instance, so the input size is 28. As there are 10 classes, output size is 10. Following Eckhardts advice once again, I chose the number of hidden nodes using the formula: floor(2 * input_size * output_size / 3), resulting in a hidden size of 186. After further experimentation, performance, both accuracy and speed, did not change much when reducing the hidden size to floor(input_size * output_size / 5) = 56, or when increasing it to input_size * output_size = 280, so I chose to stay with the originally suggested 186.

| Variable(s) | Dimensions |
| -------- | ------ |
| X<sub>t</sub> | 28x1 |
| h<sub>t</sub> | 186x1 |
| s<sub>t</sub> | 186x1 |
| W, W<sub>g</sub>, W<sub>f</sub>, W<sub>o</sub> | 186x186 |
| U, U<sub>g</sub>, U<sub>f</sub>, U<sub>o</sub>  | 186x28 | 
| b, b<sub>g</sub>, b<sub>f</sub>, b<sub>o</sub> | 186x1 |

![lstm dimensions](https://github.com/ete2njit/CS482_CNN-RNN/blob/main/resources/outputrnn.png)

The final output h<sub>28</sub> is then given as input to a fully connected neural network equivalent to matrix V of dimensionality |V| = 10\*186 such that \\
V \* h<sub>28</sub>
results in a 10x1 vector, which is then given to a softmax function to attain a final probability distribution with 10 classes.

I also make use of a 2 layer LSTM network in the performance comparison, with architecture:

![2 Layer Architecture](https://github.com/ete2njit/CS483_CNN-RNN/blob/main/resources/lstm_2_layer_architecture.png)

with slightly different dimensionalities:

![2 layer architecture dimensions](https://github.com/ete2njit/CS482_CNN-RNN/blob/main/resources/outputrnn2.png)

# Problem 2, Implementation)

[Here](https://colab.research.google.com/drive/1tf-U-YScYBok_h41_JqFSmZFpwBjQk5z?usp=sharing) is the implementation of the LSTM architecture laid out above, as well as a CNN implementation. The implemented CNN architecture can be seen at [this link](https://github.com/ete2njit/CS482_CNN-RNN/blob/main/resources/outputcnn.png).
Alternatively, the same notebook can be found under this repository's resource folder [here](https://github.com/ete2njit/CS482_CNN-RNN/blob/main/resources/CNN_RNN.ipynb)

# Problem 3, Performance

LSTM performance:
![](https://github.com/ete2njit/CS482_CNN-RNN/blob/main/resources/LSTM_performance.png)

CNN performance:
![](https://github.com/ete2njit/CS482_CNN-RNN/blob/main/resources/CNN_performance.png)

As can be seen, both models perform similarly well, with the LSTM model training slightly faster, at 13 miliseconds per step compared to the CNN-models 16 miliseconds, while the CNN model pulls slightly ahead in accuracy, at a peak and final validation accuracy of 91.3 and 90.94, respectively, whereas the single layer LSTM reaches a peak accuracy of 90.24 and a final accuracy of 89.96 on the validation set. In an effort to bring the accuracies closer together, I tried using a model with two stacked LSTM layers, which resulted in the following performance:

2 layer LSTM performance:
![](https://github.com/ete2njit/CS482_CNN-RNN/blob/main/resources/LSTM_2_layer_performance.png)

As can be seen, the accuracy of this model is slightly more competitive, but in doing so the parametercount grew from ~160k (single layer LSTM) to ~3.2m, and the time per step rose from 13 miliseconds to 37 miliseconds, meaningfully slower than the CNN model.

# Problem 4, CNN + RNN in multi-label classification)

As we can see in the performance comparisons above, CNN's seem to perform better at classifying images than LSTM neurons do. However, one thing LSTM neurons are very competent at is context evaluation. In the research paper 'CNN-RNN: A Unified Framework for Multi-label Image Classification', the researchers describe a model that aims to lean into the strengths of these two architectures. Their model utilizes a pre-trained 16 layer network as the CNN module to create an image representation. This image representation is then used by the RNN to generate a label-vector. Using this label-vector, the network calculates the distance to label embeddings and can output label probabilities accordingly. After this step, the model tries to predict another label, this time with the 'knowledge' or 'memory' of the previously predicted label, due to the nature of RNN cells. This leads to the RNN implicitly learning an attention pattern, as illustrated in the paper in figure 10. 

For the task this model was created for - multi-label prediction - the memory-like features of the RNN cell are quite useful. As the authors state, labels are not independent. It is beneficial to the model if it can find correlations, specifically co-occurence dependencies, between labels, as this feature can directly lead to improved prediction precision. If half of all labels in a dataset occur in images of the 'outside', and the other half in images of 'inside' of for example houses, predicting which of these two characteristics an image falls under can effectively cut the tasks difficulty in half.

This idea of narrowing the search space is expressed in the training section 3.4. The reasearchers note that in their training, the order in which labels are processed is according to how often they appear in the training data. By having the more common labels towards the front, the reasearchers argue that 'easier objects should be predicted first to help predict more difficult objects'. This intuition is similar to the previously mentioned reduction in search-space, making the problem 'easier' every step thanks to the learned co-occurence dependencies. 

The novelty of the proposed model is the use of 'memory' through an RNN cell to allow the model to not treat labels as independent. For the above reasons, it makes sense that such an addition would lend itself towards multi-label classification. However, in the task of single-label prediction such as in the fashion-MNIST dataset, this combination of RNN and CNN does not add much value. The main addition is the implicitly created label embedding space, which describes how closely related the model has learned labels to be. This could be used, even in a single label environment, to weigh the magnitude of errors by a more rigorous measurement than just prediction likelihood alone, but also by difference of labels. In the fashion-MNIST case, for example, of the ten labels, the labels 'Shirt' and 'T-shirt/top' are much semantically closer than 'Shirt' and 'Ankle Boot' are. The CNN RNN model in the paper, demonstrates an ability to learn label redundancies and co-occurence, which can be applied to differentiate between the severity of misclassifications by analysing the distance in label embeddings, which can be used to better train a model.


# Works Used

Eckhardt, Karsten. 'Choosing the right Hyperparameters for a simple LSTM using Keras'. Towards Data Science. Nov, 2018.
Wang, Jiang et al. 'CNN-RNN: A Unified Framework for Multi-label Image Classification'. IEEE.
