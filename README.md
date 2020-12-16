This is a simple neural network library written in C++ as part of my matura project.

There is no support for fancy stuff like autograd, convolutional layers, or even softmax or cross-entropy. Still, it does work for the MNIST dataset, the "Hello World" of neural networks, correctly predicting about 95% of the test set on a good day.
While all of this code was written by me, inspiration regarding the overall structure and naming has been taken from popular neural network libraries such as pytorch.

##### Create a data loader:
```cpp
//create a dataset from the mnist train files
data::MNIST trainSet("train");
//create a data loader from trainset with batch size = 64
data::DataLoader trainLoader(trainSet, 64);
```

##### Create a neural network:
```cpp
//create network taking 784 inputs, passing them through 3 layers and returning 10 outputs
nnet::Network net({ 
 new nnet::Linear(784,16, func::act::reLU()),
 new nnet::Linear(16, 16, func::act::reLU()),
 new nnet::Linear(16, 10, func::act::sigmoid()) },
 new func::loss::MSE()); //use mean squared error as loss function
```

##### Create an optimizer:
```cpp
//create stochastic gradient descent optimizer with learn rate=0.1 and learn rate decay speed=0.01
optim::SGD optimizer(net.layers, 0.1, 0.01);
```

##### Now use it to train the network:
```cpp
while (!trainLoader.endReached()) 
{
  //get the next 64 items from the dataset
  data::Batch batch = trainLoader.next();
  
  //reset the accumulated gradients to 0
  optimizer.zeroGrad();
  
  for (size_t i = 0; i < bat.size(); i++) 
  {
      //pass each input through the network
      net.forward(*batch[i].input);
      
      //compute the gradients with the respective label
      net.backward(*batch[i].label);
  }
  
  //update the weights with the average gradient across the batch
  optimizer.step();
}
```
