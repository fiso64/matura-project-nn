#pragma once
#include <vector>
#include "nnet.h"

#define IOPTIMIZABLE_ONLY(T) T=nnet::Linear, typename = typename std::enable_if<std::is_base_of<optim::IOptimizable, T>::value, T>::type

namespace optim
{
    //Optimizable interface: implements update() and zeroGrad(). Used by class SGD.
    class IOptimizable
    {
    public:
        virtual void update(float lr) = 0;
        virtual void zeroGrad() = 0;
    };

    //Stochastic gradient descent optimizer
    template <class IOPTIMIZABLE_ONLY(T)>
    class SGD
    {
    public:
        float initialLearningRate;
        float learningRate;
        float lrDecaySpeed;
        int callCount = 0;
        std::vector<T*> parameters;
    public:
        //IN: IOptimizable parameters, learning rate, decay speed of learning rate
        SGD(std::vector<T*> params, float lr, float decaySpeed = 0)
        {
            parameters = params;
            initialLearningRate = lr;
            learningRate = lr;
            lrDecaySpeed = decaySpeed;
        }

        //Updates each element with learning rate
        void step()
        {
            if (lrDecaySpeed != 0) {
                learningRate = initialLearningRate * (1 / (1 + sqrt(callCount * lrDecaySpeed)));
                callCount++;
            }
            for (int i = 0; i < parameters.size(); i++) {
                parameters[i]->update(learningRate);
            }
        }
        //Sets the accumulated gradient of each element to zeros
        void zeroGrad()
        {
            for (int i = 0; i < parameters.size(); i++) {
                parameters[i]->zeroGrad();
            }
        }
    };
}