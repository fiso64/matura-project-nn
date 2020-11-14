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
        //Returns a std::vector from std::vector with each element passed through forward().
        //IN:
        //OUT: new std::vector with each element newVec[i] = forward(vec[i])
        virtual Vectorf forward(Vectorf& vec)
        {
            Vectorf newVec(vec.size);
            for (size_t i = 0; i < vec.size; i++) {
                newVec[i] = forward(vec[i]);
            }
            return newVec;
        }
        //Returns a std::vector from std::vector with each element passed through backward().
        //IN:
        //OUT: new std::vector with each element newVec[i] = backward(vec[i])
        virtual Vectorf backward(Vectorf& inVec, Vectorf& inGradVec)
        {
            Vectorf newVec(inVec.size);
            for (size_t i = 0; i < inVec.size; i++) {
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
        //Return a std::vector of the unsummed losses.
        //IN: two std::vectors of equal size
        virtual Vectorf forward(Vectorf& outVec, Vectorf& labelVec) = 0;
        //Return the derivative of the loss with respect to each element in a std::vector.
        //IN: parameter std::vector, label std::vector
        //OUT: gradient std::vector with respect to parameter std::vector
        virtual Vectorf backward(Vectorf& outVec, Vectorf& labelVec) = 0;
        //Calculate the loss.
        //IN: two std::vectors of equal size
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
        public:
            float grad;
            lReLU(float gradBelowZero = 0.01F) : grad(gradBelowZero) {}
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
        };
        //standard logistic function
        class stdLogistic : public AActFunction
        {
            float forward(float x) override
            {
                return 1 / (1 + exp(-x));
            }
            float gradient(float x) override
            {
                float ex = exp(x);
                return (ex / pow((ex + 1), 2));
            }
        };
        //standard logistic function between -5 and 5, otherwise returns 0 and 1 respectively
        class stdLogisticLinearEnds : public AActFunction
        {
            float forward(float x) override
            {
                x = x / 100;
                if (x > 5) return 1;
                else if (x > -5) return 1 / (1 + exp(-x));
                else return 0;
            }
            float gradient(float x) override
            {
                x = x / 100;
                if (x > 5) return 0.0001;
                else if (x > -5) return (exp(x) / pow((exp(x) + 1), 2));
                else return -0.0001;
            }
        };
        //softmax function
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
                Vectorf expVec(vec.size);
                float expSum = 0;
                for (size_t i = 0; i < expVec.size; i++) {
                    float e = exp(vec[i]);
                    expVec[i] = e;
                    expSum += e;
                }
                Vectorf newVec(vec.size);
                for (size_t i = 0; i < vec.size; i++) {
                    newVec[i] = expVec[i] / expSum;
                }
                return newVec;
            }
            Vectorf backward(Vectorf& sums, Vectorf& sumGrad) override
            {
                Vectorf expVec(sums.size);
                float expSum = 0;
                for (size_t i = 0; i < expVec.size; i++) {
                    float e = exp(sums[i]);
                    expVec[i] = e;
                    expSum += e;
                }
                Vectorf newSumGrad(sums.size);
                for (size_t i = 0; i < sums.size; i++) {
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
    }
    //loss functions
    namespace loss
    {
        //mean squared error loss
        class MSE : public ALossFunction
        {
            Vectorf forward(Vectorf& vec1, Vectorf& vec2) override
            {
                Vectorf newVec(vec1.size);
                for (size_t i = 0; i < vec1.size; i++) {
                    newVec[i] = pow((vec1[i] - vec2[i]), 2) / (2 * vec1.size);
                }
                return newVec;
            }
            Vectorf backward(Vectorf& vec1, Vectorf& vec2) override
            {
                Vectorf newVec(vec1.size);
                for (size_t i = 0; i < vec1.size; i++) {
                    newVec[i] = (vec1[i] - vec2[i]) / vec1.size;
                }
                return newVec;
            }
        };
        //cross-entropy loss
        class CrossEntropy : public ALossFunction
        {
        public:
            CrossEntropy()
            {
                std::cout << "\n\nWARNING: " << __FUNCTION__ << " not yet implemented\n\n\n";
            }
            Vectorf forward(Vectorf& netOut, Vectorf& label) override
            {
                Vectorf newVec(netOut.size);
                for (size_t i = 0; i < netOut.size; i++) {
                    newVec[i] = -(label[i] * log(netOut[i] + 0.00001));
                }
                return newVec;
            }
            Vectorf backward(Vectorf& netOut, Vectorf& label) override
            {
                Vectorf newVec(netOut.size);
                for (size_t i = 0; i < netOut.size; i++) {
                    newVec[i] = -(label[i] / netOut[i]);
                }
                return newVec;
            }
        };
    }
    //weight initialization functions
    namespace weightInit
    {
        //Creates rows*cols matrix with 0.5
        Matrixf constInit(int rows, int cols)
        {
            float num = 0.5;
            Matrixf weightMat(rows, cols, lin::number, {num});
            return weightMat;
        }
        //Creates rows*cols matrix with a uniform distribution from -1 to 1
        Matrixf uniformInit(int rows, int cols)
        {
            Matrixf weightMat(rows, cols, lin::uniform, { -1, 1 });
            return weightMat;
        }
        //Creates rows*cols matrix with He initialization: normal distribution with mean 0 and stddev = sqrt(2/cols)
        Matrixf heInit(int rows, int cols)
        {
            float stddev = sqrt(2.0 / (double)cols);
            Matrixf weightMat(rows, cols, lin::normal, { 0, stddev });
            return weightMat;
        }
        //Creates rows*cols matrix with a normal distribution with mean 0 and stddev = sqrt(2/cols) / 2
        Matrixf heInitHalfStd(int rows, int cols)
        {
            float stddev = sqrt(2.0 / (double)cols) * 0.5;
            Matrixf weightMat(rows, cols, lin::normal, { 0, stddev });
            return weightMat;
        }
        //Creates rows*cols matrix with  Xavier initialization: normal distribution with mean 0 and stddev = sqrt(2/(cols+rows))
        Matrixf xavierInit(int rows, int cols)
        {
            float stddev = sqrt(2.0 / (((double)cols) + ((double)rows)));
            Matrixf weightMat(rows, cols, lin::normal, { 0, stddev });
            return weightMat;
        }
        Matrixf idenInit(int rows, int cols)
        {
            Matrixf weightMat = heInit(rows, cols) * 0.01;
            return weightMat + Matrixf(rows, cols, lin::identity);
        }
    }
}