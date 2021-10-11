# OCR-letters-recognition-Nerual-Network
This program is using Stochastic Gradient Descent (SGD) algorithm to build a Neural Network (NN) that could identify Optical Characters with 99% of accuracy. The Neural Network has three inner layers, which are fullyConnect layer, activation layer, loss layer. To show the accuracy of the NN, we create the Activation layer. To train the model, we use 60000 photos with their labels. 40000 photos are used in training the NN, 10000 photos are used in validating the NN, and 10000 photos are used in testing the NN.

# parameters 
To achieve 99% accuracy, we have three fully connected layers and two sigmoid layers in the neural network. The batch size is 1024, and we have 150 epochs. The learning rate is 3000. The output of the first fully connected layer and second fully connected layer is 42. 

# how it work
To implement machine learning, we use a typical model: the Neural network (NN). The basic linear model of NN is made by three main layers: which are the input layer (a1, a2, and a3 in the graph), the network layer (b1 and b2 in the graph), and the output layer (h1 and h2 in the layer). 

 ![image](https://user-images.githubusercontent.com/72623963/136716765-1bc07548-03ee-48c2-80c4-089d85a6907a.png)

In Order  to get b1 and b2, we need to multiply the weights to corresponding inputs and then add a bias to the result. In math, we could write the process to calculate b1 and b2 as matrix calculation: 
 
![image](https://user-images.githubusercontent.com/72623963/136716784-8f13a4e6-a14d-4869-a4e7-3a2241f746db.png)

Theoretically, if we could get perfect weight and bias then the b1 and b2 would have a perfect output. So, we could say that the process of achieving the best output is the process of founding the best bias and weight. Since we have the expected result b1 and b2, we can compare it to the result get from the neural network to get a loss, which is also known as the lost layer, and use the loss to revise our neural network. Due to our result is linear, we need to use an activation layer to make the result become a non-linear one.

# sources
All training photos and their labels are in "src". you can also find the specific paper about this neural network in "src".
the NN.py is the training module, and the dataPreloading.py is converting photos into 289*1's list and pack them with their labels into .npy files. 
