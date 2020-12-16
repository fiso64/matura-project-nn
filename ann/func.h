#pragma once
#include "linalg.h"

namespace func
{
    typedef linalg::Vector<float> Vectorf;
    typedef linalg::Matrix<float> Matrixf;

    //Abstract class for differentiable activation functions
    class AActFunction
    {
    public:
        //Returns the output of the activation function.
        virtual float forward(float x) = 0;
        //Returns the gradient of the activation function at x.
        virtual float gradient(float x) = 0;

        //Returns the gradient of the activation function at x multiplied by y (chain rule).
        //IN: parameter x, factor y
        virtual float backward(float x, float y)
        {
            return gradient(x) * y;
        }

        //Returns a Vectorf from Vectorf with each element passed through forward().
        //IN:
        //OUT: new std::vector with each element newVec[i] = forward(vec[i])
        virtual Vectorf forward(Vectorf& vec)
        {
            Vectorf newVec(vec.size());
            for (size_t i = 0; i < vec.size(); i++) {
                newVec[i] = forward(vec[i]);
            }
            return newVec;
        }

        //Returns a Vectorf with each element passed through backward().
        //IN:
        //OUT: new Vectorf with each element newVec[i] = backward(vec[i])
        virtual Vectorf backward(Vectorf& inVec, Vectorf& inGradVec)
        {
            Vectorf newVec(inVec.size());
            for (size_t i = 0; i < inVec.size(); i++) {
                newVec[i] = backward(inVec[i], inGradVec[i]);
            }
            return newVec;
        }

        float operator () (float x)
        {
            return forward(x);
        }
        Vectorf operator () (Vectorf& vec)
        {
            return forward(vec);
        }
    };
    //Abstract class for differentiable loss functions
    class ALossFunction
    {
    public:
        //Return a vector of the unsummed losses.
        //IN: two vectors of equal size
        virtual Vectorf forward(Vectorf& outVec, Vectorf& labelVec) = 0;

        //Return the derivative of the loss with respect to each element in a vector.
        //IN: parameter vector, label vector
        //OUT: gradient vector with respect to parameter vector
        virtual Vectorf backward(Vectorf& outVec, Vectorf& labelVec) = 0;

        //Calculate the loss.
        //IN: two vectors of equal size
        //OUT: the loss
        virtual float numericLoss(Vectorf& outVec, Vectorf& labelVec)
        {
            return forward(outVec, labelVec).sum();
        }

        float operator () (Vectorf& vec1, Vectorf& vec2)
        {
            return numericLoss(vec1, vec2);
        }
    };

    //activation functions
    namespace act
    {
        //rectified linear unit
        class reLU : public AActFunction
        {
            float forward(float x) override
            {
                if (x > 0) return x;
                else return 0;
            }
            float gradient(float x) override
            {
                if (x >= 0) return 1;
                else return 0;
            }
        };

        //leaky rectified linear unit
        class lReLU : public AActFunction
        {
            float forward(float x) override
            {
                if (x > 0) return x;
                else return grad * x;
            }
            float gradient(float x) override
            {
                if (x >= 0) return 1;
                else return grad;
            }
        public:
            float grad;
            lReLU(float grad_ = 0.01F) { grad = grad_; }
        };

        //sigmoid function
        class sigmoid : public AActFunction
        {
            float forward(float x) override
            {
                x = x / squeeze;
                return 1 / (1 + exp(-x));
            }
            float gradient(float x) override
            {
                x = x / squeeze;
                float ex = exp(x);
                return (ex / pow((ex + 1), 2));
            }
        public:
            float squeeze;
            sigmoid(float squeeze_ = 50) { squeeze = squeeze_; }
        };

        //standard logistic function between -5 and 5, otherwise returns 0 and 1 respectively
        class logisticLinearEnds : public AActFunction
        {
            float forward(float x) override
            {
                x = x / squeeze;
                if (x > 5) return 1;
                else if (x > -5) return 1 / (1 + exp(-x));
                else return 0;
            }
            float gradient(float x) override
            {
                x = x / squeeze;
                if (x > 5) return 0.0001;
                else if (x > -5) return (exp(x) / pow((exp(x) + 1), 2));
                else return -0.0001;
            }
        public:
            float squeeze;
            logisticLinearEnds(float squeeze_ = 100) { squeeze = squeeze_; }
        };

        //softmax function, not working as of now
        class softMax : public AActFunction
        {
        public:
            softMax()
            {
                std::cout << "\n\nWARNING: " << __FUNCTION__ << " not yet implemented\n\n\n";
            }

            float forward(float x) override
            {
                if (x > 0) return x;
                else return 0;
            }
            float backward(float x, float y) override
            {
                if (x >= 0) return y;
                else return 0;
            }
            Vectorf forward(Vectorf& vec) override
            {
                Vectorf expVec(vec.size());
                float expSum = 0;
                for (size_t i = 0; i < expVec.size(); i++) {
                    float e = exp(vec[i]);
                    expVec[i] = e;
                    expSum += e;
                }
                Vectorf newVec(vec.size());
                for (size_t i = 0; i < vec.size(); i++) {
                    newVec[i] = expVec[i] / expSum;
                }
                return newVec;
            }
            Vectorf backward(Vectorf& sums, Vectorf& sumGrad) override
            {
                Vectorf expVec(sums.size());
                float expSum = 0;
                for (size_t i = 0; i < expVec.size(); i++) {
                    float e = exp(sums[i]);
                    expVec[i] = e;
                    expSum += e;
                }
                Vectorf newSumGrad(sums.size());
                for (size_t i = 0; i < sums.size(); i++) {
                    newSumGrad[i] = ((expSum * expVec[i]) / pow((expSum + expVec[i]), 2)) * sumGrad[i];
                }
                return newSumGrad;
            }
        };

        //sine activation
        class sinAct : public AActFunction
        {
            float forward(float x) override
            {
                return (1 + sin(x)) / 2;
            }
            float gradient(float x) override
            {
                return cos(x) / 2;
            }
        };

        //exponential activation
        class expAct : public AActFunction
        {
            float forward(float x) override
            {
                return exp(x);
            }
            float gradient(float x) override
            {
                return exp(x);
            }
        };

        //linear activation
        class linear : public AActFunction
        {
            float forward(float x) override
            {
                return grad * x;
            }
            float gradient(float x) override
            {
                return grad;
            }
        public:
            float grad;
            linear(float grad_ = 0.01F) { grad = grad_; }
        };
    }

    //loss functions
    namespace loss
    {
        //mean squared error loss
        class MSE : public ALossFunction
        {
            Vectorf forward(Vectorf& outs, Vectorf& labels) override
            {
                Vectorf newVec(outs.size());
                for (size_t i = 0; i < outs.size(); i++) {
                    newVec[i] = pow((outs[i] - labels[i]), 2) / (2 * outs.size());
                }
                return newVec;
            }
            Vectorf backward(Vectorf& outs, Vectorf& labels) override
            {
                Vectorf newVec(outs.size());
                for (size_t i = 0; i < outs.size(); i++) {
                    newVec[i] = (outs[i] - labels[i]) / outs.size();
                }
                return newVec;
            }
        };

        //loss for logistic regression
        class Logistic : public ALossFunction
        {
            Vectorf forward(Vectorf& outs, Vectorf& labels) override
            {
                Vectorf newVec(outs.size());
                for (size_t i = 0; i < outs.size(); i++) {
                    newVec[i] = labels[i] * log(outs[i]) + (1 - labels[i]) * log(1 - outs[i]);
                }
                return newVec;
            }
            Vectorf backward(Vectorf& outs, Vectorf& labels) override
            {
                Vectorf newVec(outs.size());
                for (size_t i = 0; i < outs.size(); i++) {
                    newVec[i] = - (outs[i] - labels[i]) / (outs[i] * (1 - outs[i]));
                }
                newVec.print();
                return newVec;
            }
        };

        //cross-entropy loss, not done yet
        class CrossEntropy : public ALossFunction
        {
        public:
            CrossEntropy()
            {
                std::cout << "\n\nWARNING: " << __FUNCTION__ << " not yet implemented\n\n\n";
            }
            Vectorf forward(Vectorf& netOut, Vectorf& label) override
            {
                Vectorf newVec(netOut.size());
                for (size_t i = 0; i < netOut.size(); i++) {
                    newVec[i] = -(label[i] * log(netOut[i] + 0.00001));
                }
                return newVec;
            }
            Vectorf backward(Vectorf& netOut, Vectorf& label) override
            {
                Vectorf newVec(netOut.size());
                for (size_t i = 0; i < netOut.size(); i++) {
                    newVec[i] = -(label[i] / netOut[i]);
                }
                return newVec;
            }
        };
    }

    //weight initialization functions
    namespace weightInit
    {
        //Creates rows*cols matrix with ones
        Matrixf constInit(int rows, int cols)
        {
            float num = 1;
            Matrixf weightMat(rows, cols, linalg::number, {num});
            return weightMat;
        }

        //Creates rows*cols matrix with a uniform distribution from -1 to 1
        Matrixf uniformInit(int rows, int cols)
        {
            Matrixf weightMat(rows, cols, linalg::uniform, { -1, 1 });
            return weightMat;
        }

        //Creates rows*cols matrix with He initialization: normal distribution with mean 0 and stddev = sqrt(2/cols)
        Matrixf heInit(int rows, int cols)
        {
            float stddev = sqrt(2.0 / (double)cols);
            Matrixf weightMat(rows, cols, linalg::normal, { 0, stddev });
            return weightMat;
        }

        //Creates rows*cols matrix with a normal distribution with mean 0 and stddev = sqrt(2/cols) / 2
        Matrixf heInitHalfStd(int rows, int cols)
        {
            float stddev = sqrt(2.0 / (double)cols) * 0.5;
            Matrixf weightMat(rows, cols, linalg::normal, { 0, stddev });
            return weightMat;
        }

        //Creates rows*cols matrix with  Xavier initialization: normal distribution with mean 0 and stddev = sqrt(2/(cols+rows))
        Matrixf xavierInit(int rows, int cols)
        {
            float stddev = sqrt(2.0 / (((double)cols) + ((double)rows)));
            Matrixf weightMat(rows, cols, linalg::normal, { 0, stddev });
            return weightMat;
        }

        Matrixf idenInit(int rows, int cols)
        {
            Matrixf weightMat = heInit(rows, cols) * 0.01;
            return weightMat + Matrixf(rows, cols, linalg::identity);
        }

        Matrixf zeroInit(int rows, int cols)
        {
            return Matrixf(rows, cols, linalg::zeros);
        }
    }
}