# OCR-Letters-Recognition-Nerual-Network
This program is using **Stochastic Gradient Descent** (**SGD**) algorithm to build a **Neural Network** (**NN**) that could identify Optical Characters with 99% of accuracy. The Neural Network has three inner layers, which are **fullyConnect layer, activation layer, loss layer**. To show the accuracy of the **NN**, we create the **Activation layer**. To train the model, we use 60000 photos with their **labels**. 40000 photos are used in training the **NN**, 10000 photos are used in validating the **NN**, and 10000 photos are used in testing the **NN**.

# How to tune the NN 
At first, we try single fully connected network and get 92.67% accuracy in 150 epochs with 0.0425 loss. In the third epoch it achieves 60.20% accuracy. By adding one fully connect layer, the accuracy after 150 epochs is 97.6%. However, it takes 15 epochs for the neural network to get to 60% accuracy. 

 When we add the third layer, the final accuracy is 97.97 %, which does not improve much caused by gradient vanishing: when the input of sigmoid layer is small enough or large enough, the output would have only subtle change than the output from a close input.  

There are many solutions to this problem, like replacing activation layer with function like ReLu, LeakyRelu, elu. One other efficient way is to use Cross-Entropy Loss function The formula of this function would like is 

![image](https://user-images.githubusercontent.com/72623963/137162056-c27605f1-eff0-4069-a521-ef53428e4614.png)


The main reason why we use is its derivative is 
 
![image](https://user-images.githubusercontent.com/72623963/137162082-d7b69637-f90f-4bfd-af4f-bbd0b44ffeab.png)


which would become h (theta, x) after it times the sigmoid function's derivative.

Beside adding more fully connected layers, we could adjust parameters in the NN to revise it. ------

To achieve 99% accuracy, we have three **fully connected layers and two sigmoid layers** in the **neural network**. The **batch size** is 1024, and we have 150 **epochs**. The **learning rate** is 3000. The output of the first **fully connected layer** and second **fully connected layer** is 42.
 
# How it work
To implement machine learning, we use a typical model: the **Neural network** (NN). The basic linear model of NN is made by three main layers: which are the **input layer** (a1, a2, and a3 in the graph), the **network layer** (b1 and b2 in the graph), and the **output layer** (h1 and h2 in the layer). 

 ![image](https://user-images.githubusercontent.com/72623963/136716765-1bc07548-03ee-48c2-80c4-089d85a6907a.png)

In Order to get b1 and b2, we need to multiply the **weights** to corresponding inputs and then add a **bias** to the result. In math, we could write the process to calculate b1 and b2 as matrix calculation: 
 
![image](https://user-images.githubusercontent.com/72623963/136716784-8f13a4e6-a14d-4869-a4e7-3a2241f746db.png)

Theoretically, if we could get perfect **weight** and **bias** then the b1 and b2 would have a perfect output. So, we could say that the process of achieving the best output is the process of founding the best **bias** and **weight**. Since we have the expected result b1 and b2, we can compare it to the result get from the neural network to get a **loss**, which is also known as the **lost layer**, and use the loss to revise our **neural network**. Due to our result is linear, we need to use an **activation layer** to make the result become a non-linear one.

# Sources
All training photos and their labels are in "src". you can also find the specific paper about this neural network in "src".
the NN.py is the training module, and the dataPreloading.py is converting photos into 289*1's list and pack them with their labels into .npy files. 
