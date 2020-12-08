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
    class ILayer : public optim::IOptimizable
    {
    public:
        int inSize; //size of the layer's input vector
        int outSize; //size of the layer's output vector
        Vectorf outs; //the outputs of this layer
        Vectorf* prevOuts; //the outputs of the previous layer
        Matrixf weights; //the layer's weight matrix
        Vectorf biases; //the layer's bias weights
        virtual Vectorf forward(Vectorf& inVec) = 0;
        virtual Vectorf backward(Vectorf& outGrad) = 0;
    };
    
    //Linear network layer
    class Linear : public ILayer
    {
    public:
        Vectorf sums; //the weighted sums of each neuron
        Matrixf weightsGradSum; //the sum of multiple weight gradients (for SGD)
        Vectorf biasesGradSum; //the sum of multiple bias gradients
        bool bias = false; //use bias neuron
        func::AActFunction* actFunc; //the activation function used by each neuron
        int batchSize = 0; //the current batch size to determine how large the step for SGD should be
    public:
        //IN: amount of inputs, amount of outputs, activation function, weight initialization function
        template <class T>
        Linear(int inChan, int outChan, T, bool bias_ = false, std::function<Matrixf(int, int)> weightInit = func::weightInit::heInitHalfStd)
        {
            //static_assert(std::is_base_of<func::AActFunction, T>::value);
            bias = bias_;
            actFunc = new T;
            inSize = inChan;
            outSize = outChan;
            sums = Vectorf(outChan);
            outs = Vectorf(outChan);
            if(weightInit != NULL) weights = weightInit(outChan, inChan); //initialize weights
            weightsGradSum = Matrixf(outChan, inChan, lin::zeros);
            biases = Vectorf(outChan, lin::zeros);
            biasesGradSum = Vectorf(outChan, lin::zeros);
        }
        template <class T>
        Linear(int inChan, int outChan, T* fptr, bool bias_ = false, std::function<Matrixf(int, int)> weightInit = func::weightInit::heInitHalfStd)
        {
            //static_assert(std::is_base_of<func::AActFunction, T>::value);
            bias = bias_;
            actFunc = fptr;
            inSize = inChan;
            outSize = outChan;
            sums = Vectorf(outChan);
            outs = Vectorf(outChan);
            if (weightInit != NULL) weights = weightInit(outChan, inChan);
            weightsGradSum = Matrixf(outChan, inChan, lin::zeros);
            biases = Vectorf(outChan, lin::zeros);
            biasesGradSum = Vectorf(outChan, lin::zeros);
        }
        ~Linear()
        {
            delete actFunc;
        }

        //Compute the outputs of the layer from the inputs.
        //IN: the outputs of the previous layer
        //OUT: the outputs of this layer
        Vectorf forward(Vectorf& inVec) override
        {
            //multiply the inputs by the weight matrix to get the weighted sums.
            //Save them to use in backward pass.
            sums = weights * inVec; 
            //pass the (biased) weighted sums thru the act. func to get the outputs
            if (bias) {
                Vectorf vec = sums + biases;
                outs = actFunc->forward(vec);
            }
            else outs = actFunc->forward(sums);

            return outs;
        }

        //compute the gradient w.r.t the weights, add the gradient to weightsGradSum, and prepare sumGrad for backward() of the previous layer.
        //IN: gradient w.r.t the outputs, returned from backward() of the next layer
        //OUT: gradient w.r.t the outputs of the previous layer
        Vectorf backward(Vectorf& outGrad) override
        {
            //we need to "go back" for each function we used in the net. The derivative of the loss function has already been calculated in the net's backward().

            //from the recursive definition of the gradient of loss w.r.t the sums of a layer: compute d(outs)/d(sums) first, 
            //which is the derivative of the activation function:
            Vectorf sumGrad = actFunc->backward(sums, outGrad);
            //to then get the gradient w.r.t the weights, the resulting sumGrad needs to be multiplied by d(sums)/d(weights) (chain rule),
            //which is equal to the outputs of the previous layer:
            Matrixf weightsGrad = sumGrad.transposed() * (*prevOuts).asMatrix();
            //add the weight gradient to the sum to then use it in SGD:
            weightsGradSum += weightsGrad;
            if (bias) biasesGradSum += actFunc->backward(biases, outGrad);
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
            if (bias) biases -= biasesGradSum * (lr / batchSize);
        }

        //Set weightsGradSum to zero.
        //
        void zeroGrad() override
        {
            //zero the weight gradient sum. The update step should be independent from batch to batch in SGD.
            weightsGradSum *= 0;
            if (bias) biasesGradSum *= 0;
            batchSize = 0;
        }
    };

    class Network
    {
    protected:
        func::ALossFunction* lossFuncPtr;
    public:
        std::vector<ILayer*> layers; //each layer in the network
        func::ALossFunction& lossFunc; //the loss function
        Vectorf* output; //the outputs of the last layer
    public:
        //IN: Any amount of layers, loss function
        template <class T, class U>
        Network(std::initializer_list<T*> lrs, U& lossFnc, std::function<Matrixf(int, int)> weightInit = NULL, float weightsMult = 1)
            : lossFunc(lossFnc) 
        {
            lossFuncPtr = new U;
            layers.assign(lrs.begin(), lrs.end());

            //set prevOuts of each layer to point to the outputs of the previous layers
            for (auto it = layers.begin() + 1; it != layers.end(); it++) {
                (*it)->prevOuts = (&(*(it - 1))->outs);
            }
            layers[0]->prevOuts = new Vectorf(layers[0]->inSize);

            //intialize weights
            if (weightInit != NULL) {
                for (auto l : layers) {
                    l->weights = weightInit(l->outSize, l->inSize);
                    if (weightsMult != 1) l->weights *= weightsMult;
                }
            }

            output = &(layers[layers.size() - 1]->outs);
        }
        template <class T, class U>
        Network(std::initializer_list<T*> lrs, U* lossFncPtr, std::function<Matrixf(int, int)> weightInit = NULL, float weightsMult= 1)
            : lossFunc(*lossFncPtr)
        {
            lossFuncPtr = lossFncPtr;
            layers.assign(lrs.begin(), lrs.end());

            //set prevOuts to point to the outputs of the previous layers
            for (auto it = layers.begin() + 1; it != layers.end(); it++) {
                (*it)->prevOuts = (&(*(it - 1))->outs);
            }
            layers[0]->prevOuts = new Vectorf(layers[0]->inSize);
            
            //intialize weights
            if (weightInit != NULL) {
                for (auto l : layers) {
                    l->weights = weightInit(l->outSize, l->inSize);
                    if (weightsMult != 1) l->weights *= weightsMult;
                }
            }

            output = &(layers[layers.size() - 1]->outs);
        }
        ~Network()
        {
            delete lossFuncPtr;
            delete layers[0]->prevOuts;
            for (int i = 0; i < layers.size(); i++) {
                delete layers[i];
            }
        }

        //Propagate forward with an input vector; call forward() on each layer.
        //IN: a Vectorf of inputs to the network
        void forward(Vectorf input)
        {
            (*layers[0]->prevOuts) = input;
            for (int i = 0; i < layers.size(); i++) {
                input = layers[i]->forward(input);
            }
        }

        //Propagate backward with a label vector; call backward() on each layer.
        //IN: a Vectorf of labels
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