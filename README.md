# OCR-letters-recognition-Nerual-Network
this program is using Stochastic Gradient Descent (SGD) algorithm to build a Neural Network (NN) that could identify Optical Characters with 99% of accuracy. The Neural Network has three inner layers, which are fullyConnect layer, activation layer, loss layer. To show the accuracy of the NN, we create the Activation layer. To train the model, we use 60000 photos with their labels. 40000 photos are used in training the NN, 10000 photos are used in validating the NN, and 10000 photos are used in testing the NN.
# how it work
To implement machine learning, we use a typical model: the Neural network (NN). The basic linear model of NN is made by three main layers: which are the input layer (a1, a2, and a3 in the graph), the network layer (b1 and b2 in the graph), and the output layer (h1 and h2 in the layer). 

 ![image](https://user-images.githubusercontent.com/72623963/136716765-1bc07548-03ee-48c2-80c4-089d85a6907a.png)

In Order  to get b1 and b2, we need to multiply the weights to corresponding inputs and then add a bias to the result. In math, we could write the process to calculate b1 and b2 as matrix calculation: 
 
![image](https://user-images.githubusercontent.com/72623963/136716784-8f13a4e6-a14d-4869-a4e7-3a2241f746db.png)

Theoretically, if we could get perfect weight and bias then the b1 and b2 would have a perfect output. So, we could say that the process of achieving the best output is the process of founding the best bias and weight.  -=---

