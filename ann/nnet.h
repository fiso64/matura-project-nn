#pragma once
#include <iostream>
#include <string>
#include <algorithm>
#include <cmath>
#include <vector>
#include <functional>
#include <stdarg.h>
#include <random>
#include <map>
#include <fstream>
#include <iterator>
#include <windows.h>
#include <intrin.h>
#include <chrono>
#include <stdexcept>
#include <cassert>

#include "linalg.h"
#include "func.h"
#include "optim.h"

#define ILAYER_ONLY(T) T=Linear, typename = typename std::enable_if<std::is_base_of<ILayer, T>::value, T>::type

namespace nnet
{
    namespace lin = linalg;
    typedef lin::Vector<float> Vectorf;
    typedef lin::Matrix<float> Matrixf;

    //Layer interface: implements forward() and backward(). Used by class Network.
    class ILayer
    {
    public:
        int inSize; //size of the layer's input std::vector
        int outSize; //size of the layer's output std::vector
        virtual Vectorf forward(Vectorf& inVec) = 0;
        virtual Vectorf backward(Vectorf& outGrad) = 0;
    };
    
    //Linear network layer
    class Linear : public ILayer, public optim::IOptimizable
    {
    public:
        Vectorf sums; //the weighted sums of each neuron
        Vectorf outs; //the outputs of each neuron
        Matrixf weights; //the layer's weight matrix
        Matrixf weightsGrad; //the gradient of the weight matrix
        Matrixf weightsGradSum; //the sum of multiple weight gradients (for SGD)
        func::AActFunction* actFunc; //the activation function used by each neuron
        int batchSize = 0; //the current batch size to determine how large the step for SGD should be

        Vectorf* prevOuts; //the outputs of the previous layer
    public:
        //IN: amount of inputs, amount of outputs, activation function, weight initialization function
        Linear(int inChan, int outChan, func::AActFunction* actFnc, std::function<Matrixf(int, int)> weightInit = NULL)
            : actFunc(actFnc)
        {
            inSize = inChan;
            outSize = outChan;
            sums = Vectorf(outChan);
            outs = Vectorf(outChan);
            if(weightInit != NULL) weights = weightInit(outChan, inChan);
            weightsGrad = Matrixf(outChan, inChan);
            weightsGradSum = Matrixf(outChan, inChan, lin::zeros);
        }
        ~Linear()
        {
            delete &actFunc;
        }

        //Compute the outputs of the layer from the inputs.
        //IN: the outputs of the previous layer
        //OUT: the outputs of this layer
        Vectorf forward(Vectorf& inVec) override
        {
            //multiply the inputs by the weight matrix to get the weighted sums.
            //Save them to use in backward pass.
            sums = weights * inVec; 
            //pass the weighted sums thru the act. func to get the outputs:
            outs = actFunc->forward(sums);
            return outs;
        }
        //compute the gradient w.r.t the weights, add the gradient to weightsGradSum, and prepare sumGrad for backward() of the previous layer.
        //IN: gradient w.r.t the outputs, returned from backward() of the next layer
        //OUT: gradient w.r.t the outputs of the previous layer
        Vectorf backward(Vectorf& outGrad) override
        {
            //from the recursive definition of the gradient of loss w.r.t the sums of a layer, compute d(outs)/d(sums) first, 
            //which is the derivative of the activation function:
            Vectorf sumGrad = actFunc->backward(sums, outGrad);
            //to then get the gradient w.r.t the weights, the resulting sumGrad needs to be multiplied by d(sums)/d(weights) (chain rule),
            //which is equal to the outputs of the previous layer:
            weightsGrad = sumGrad.transposed() * (*prevOuts).asMatrix();
            //add the weight gradient to the sum to then use it in SGD:
            weightsGradSum += weightsGrad;
            batchSize++;
            //to get the gradient w.r.t the outputs of the previous layer, multiply sumGrad by d(sums)/d(outs-1) (=weights of the previous layer).
            //The transpose appears because the gradient is computed backwards.
            Vectorf newOutGrad = weights.transposed() * sumGrad;
            return newOutGrad;
        }
        //Update the weights of the layer with the negative of weightsGradSum (accumulated during backward() calls) times a learning rate (SGD).
        //IN: learning rate
        void update(float lr) override
        {
            //update the weights with the negative of the weight gradient times the learning rate. 
            //divide by batch size because the weight update step size should be independent from batch size
            weights -= weightsGradSum * (lr / batchSize);
        }
        //Set weightsGradSum to zero.
        //
        void zeroGrad() override
        {
            //zero the weight gradient sum. The update step should be independent from batch to batch in SGD.
            weightsGradSum *= 0;
            batchSize = 0;
        }
    };
    template <class ILAYER_ONLY(T)>
    class Network
    {
    public:
        std::vector<T*> layers; //each layer in the network
        func::ALossFunction& lossFunc; //the loss function
        Vectorf* output; //the outputs of the last layer
    public:
        //IN: Any amount of layers, loss function, weight initialization function
        Network(std::initializer_list<T*> lrs, func::ALossFunction& lossFnc, std::function<Matrixf(int, int)> weightInit = func::weightInit::heInit)
            : lossFunc(lossFnc)
        {
            //add the layers to the std::vector and initialize their weights
            for (auto it = lrs.begin(); it != lrs.end(); it++) { 
                (*it)->weights = weightInit((*it)->outSize, (*it)->inSize);
                layers.push_back(*it);     
            }
            //set prevOuts to point to the outputs of the previous layers
            for (auto it = layers.begin() + 1; it != layers.end(); it++) { 
                (*it)->prevOuts = (&(*(it - 1))->outs);
            }
            layers[0]->prevOuts = new Vectorf(layers[0]->inSize);
            output = &(layers[layers.size() - 1]->outs);
        }
        ~Network()
        {
            delete &lossFunc;
            delete layers[0]->prevOuts;
            for (int i = 0; i < layers.size(); i++) {
                delete layers[i];
            }
        }

        //Propagate forward with an input std::vector; call forward() on each layer.
        //IN: a std::vector of inputs to the network
        void forward(Vectorf inputVec)
        {
            (*layers[0]->prevOuts) = inputVec;
            for (int i = 0; i < layers.size(); i++) {
                inputVec = layers[i]->forward(inputVec);
            }
        }
        //Propagate backward with a label std::vector; call backward() on each layer.
        //IN: a std::vector of labels
        void backward(Vectorf& label)
        {
            //compute the gradient of the loss w.r.t the outputs of the last layer
            Vectorf outGrad = lossFunc.backward(*output, label);
            for (int i = layers.size() - 1; i >= 0; i--) {
                //compute the gradient of the loss w.r.t the outputs of each layer
                //compute the weight gradients for each layer
                outGrad = layers[i]->backward(outGrad);
            }
        }
    };
}