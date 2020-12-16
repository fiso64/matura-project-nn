#include <iostream>
#include <locale>
#include <sstream>
#include <codecvt>
#include <thread>
#include <iomanip>
#include <vector>
#include <fstream>
#include <string>
#include <array>
#include <iterator>

#include "nnet.h"
#include "optim.h"
#include "func.h"
#include "data.h"
#include "linalg.h"
#include "helpers.h"
using namespace std;

int main()
{
    //create datasets from the mnist files if found in the default directory
    //if files are not found, enter the full paths to the mnist images and labels:
    //data::MNIST dataSet("C:\\your\\image\\path", "C:\\your\\label\\path");
    data::MNIST trainSet("train"); 
    data::MNIST testSet("test");

    //create a data loaders from the datasets
    data::DataLoader trainLoader(trainSet, 64); //batch size = 64
    data::DataLoader testLoader(testSet);

    //create network taking 784 inputs, passing them through 5 layers and returning 10 outputs
    nnet::Network net({ 
        new nnet::Linear(784,16, func::act::reLU()),
        new nnet::Linear(16, 16, func::act::reLU()),
        new nnet::Linear(16, 16, func::act::reLU()),
        new nnet::Linear(16, 16, func::act::reLU()),
        new nnet::Linear(16, 10, func::act::sigmoid()) },
        new func::loss::MSE()); //use mean squared error as loss function

    //create stochastic gradient descent optimizer with learn rate=0.1 and learn rate decay speed=0.001
    optim::SGD optimizer(net.layers, 0.1, 0.01);


    //#########################
    //### train the network ### 
    //#########################

    //uncomment for a better training loop:
    //helpers::trainAndTestNet(net, optimizer, trainLoader, testLoader, 1 /*set verbosity*/);

    cout << "started training" << endl;
    for (int epoch = 1; epoch <= 2; epoch++) 
    {
        float avgLoss = 0;
        for (int i = 0; !trainLoader.endReached(); i++) 
        {
            //zero the accumulated weight gradient sums
            optimizer.zeroGrad();

            //obtain next batch
            data::Batch batch = trainLoader.next();

            //for each element in the batch...
            for (auto element : batch) 
            {
                //...propagate forward
                net.forward(*element.input);

                //and propagate backward 
                net.backward(*element.label);

                avgLoss += net.lossFunc(*net.output, *element.label);
            }
            //update the weights with the average weight gradient
            optimizer.step();

            if (i % 100 == 0) {
                cout << "Average loss: " << avgLoss / (100 * batch.size()) << endl;
                avgLoss = 0;
            }
        }
        cout << "epoch " << epoch << " complete." << endl;
    }


    //#########################
    //### test the network #### 
    //#########################

    //get all 10000 images from the test set
    data::Batch batch = testLoader.all();

    int numRight = 0;
    for (auto element : batch) //for each element in the batch
    {
        //propagate forward to get the output
        net.forward(*element.input);

        //get the network's prediction and label digit
        int predDigit = distance((*net.output).nums.begin(), max_element((*net.output).nums.begin(), (*net.output).nums.end()));
        int labelDigit = distance((*element.label).nums.begin(), max_element((*element.label).nums.begin(), (*element.label).nums.end()));

        //compare the net’s predicted digit with the label
        if (predDigit == labelDigit) numRight++;
    }

    float rate = (numRight * 100.0) / batch.size();
    cout << "Correctly predicted " << rate << "%" << endl;

}


//TODO: Add include guards...

//TODO: Calling forward & backward on batches:                                   
//TODO: when computing for a batch, create a batch_size * features_size          
//TODO: matrix of inputs and multiply it by the weight mat (for vectorization)   
//TODO: do similarly for backprop                                                

//TODO: normalize + scale mnist
//TODO: test on custom images
//TODO: Make SGD an actual optimizer, why is the code inside the Layer class?
//TODO: CrossEntropyLoss & softMax & Momentum

//TODO: Conv layers
//TODO: Autograd
//TODO: Maybe make this work in a python module