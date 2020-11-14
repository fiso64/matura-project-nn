# matura-project-nn
A simple neural network library written in C++ as part of my matura project.

There is no support for fancy stuff like LSTMs, convolutional layers or even softmax and cross-entropy. Still, it does work for the MNIST dataset, the "Hello World" of neural networks, correctly predicting about 90% of the test set on a good day.
While all of this code is written by me, some inspiration regarding the overall structure and naming has been taken from popular neural network libraries such as pytorch.

### Some examples how one would use this "library"<br/>
##### Create a data loader:
```cpp
//create a dataset from the mnist train files
data::MNIST trainSet(DATA_DIR + "train-images.idx3-ubyte", DATA_DIR + "train-labels.idx1-ubyte");
//create a data loader from trainset with batch size = 64
data::DataLoader trainLoader(trainSet, 64);
```
 
##### Define loss function and weight initialization:
```cpp
//initialize weights with He Normal
auto weightInitFunc = func::weightInit::heInitHalfStd;
//use mean squared error as loss function
func::loss::MSE lossFunc;
```

##### Create a neural network:
```cpp
//network taking 784 inputs, passing them through 5 layers and returning 10 outputs
nnet::Network<nnet::Linear> net ({ 
  new nnet::Linear(784,16, (new func::act::reLU())),
  new nnet::Linear(16, 16, (new func::act::reLU())),
  new nnet::Linear(16, 16, (new func::act::reLU())),
  new nnet::Linear(16, 16, (new func::act::reLU())),
  new nnet::Linear(16, 10, (new func::act::stdLogisticLinearEnds())) }, 
  lossFunc, weightInitFunc);
```

##### Create an optimizer:
```cpp
//create stochastic gradient descent optimizer with learn rate=0.1 and learn rate decay speed=0.001
optim::SGD<nnet::Linear> optimizer(net.layers, 0.1, 0.001);
```

##### Now use it to train the network:
```cpp
while (!trainLoader.endReached) 
{
  //get the next 64 items from the dataset
  data::Batch batch = trainLoader.next();
  
  //reset the accumulated gradients to 0
  optimizer.zeroGrad();
  
  for (size_t i = 0; i < bat.size(); i++) {
      //pass each input through the network
      net.forward(*batch[i].input);
      //compute the gradients with the respective label
      net.backward(*batch[i].label);
  }
  
  //update the weights with the average gradient across the batch
  optimizer.step();
}
```
