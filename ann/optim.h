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
        linalg::Matrix<float> weights;
        virtual void update(float lr) = 0;
        virtual void zeroGrad() = 0;
    };

    //Stochastic gradient descent optimizer (so far it's not much of an optimizer, the steps are still done on the layers)
    class SGD
    {
    public:
        float initialLearnRate;
        float learnRate;
        float learnRateDecaySpeed;
        int callCount = 0;
        std::vector<IOptimizable*> parameters;
    public:
        //IN: IOptimizable parameters, learning rate, decay speed of learning rate
        template <class T>
        SGD(std::vector<T*> params, float lr, float decaySpeed = 0)
        {
            parameters.assign(params.begin(), params.end());
            initialLearnRate = lr;
            learnRate = lr;
            learnRateDecaySpeed = decaySpeed;
        }

        //Updates each element with learning rate
        void step()
        {
            if (learnRateDecaySpeed != 0) {
                learnRate = initialLearnRate * (1 / (1 + sqrt(callCount * learnRateDecaySpeed)));
                callCount++;
            }
            for (int i = 0; i < parameters.size(); i++) {
                parameters[i]->update(learnRate);
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