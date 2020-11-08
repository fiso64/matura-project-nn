//The following is an implementation of a simple neural network for MNIST number classification. 
//As you will be probably able to tell, I don't know shit about C++, and I started this project with very little prior experience.
//Still I chose to do it in C++, to "learn a cool new language", and considering the short amount of time I had and the immensity of C++, this now seems to be 
//nothing but an excuse for my apparent masochistic tendencies. Really, all I learned here is to be mortally afraid of C++.

// “Life is hard without a garbage collector” - Gandhi

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

#include "linalg.h" //A linear algebra library I made for another project. Because why use something like eigen when yours has so many more "features"?
using namespace std; //this is good practice
namespace lin = linalg;

struct InputLabelPair;
typedef lin::Vector<float> Vectorf;
typedef lin::Matrix<float> Matrixf;
typedef vector<InputLabelPair> Batch;

class Timer
{
public:
    chrono::steady_clock::time_point t1;
    chrono::steady_clock::time_point t2;
    long long duration = 0;
public:
    Timer() {
        t1 = chrono::high_resolution_clock::now();
    }
    void start() {
        t1 = chrono::high_resolution_clock::now();
    }
    void stop(string str = "", bool print = true) {
        t2 = chrono::high_resolution_clock::now();
        duration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count(); 
        if (print) 
            cout << endl << "TIME ELAPSED: " << duration << " microseconds = " 
            << duration/1000000.0 << " seconds. " << str;
        t1 = chrono::high_resolution_clock::now();
    }
};

struct InputLabelPair
{
    Vectorf* input;
    Vectorf* label;
};
class DataSet
{

};
class MNIST : public DataSet
{

};
class DataLoader
{
public:
    string inputPath;
    string labelPath;
    ifstream inputifs;
    ifstream labelifs;
    vector<Vectorf> inputData;
    vector<Vectorf> labelData;

    int curPos = 0;
    int totalNumItems;
    int cols, rows, inputSize;
    int labelSize = 10;
    bool shuffled;
    bool endReached = false;
public:
    DataLoader(string ipath, string lpath, bool shuffle = true) //read inputs and labels onto bytearrays, shuffle numbers, do stuff
    {
        shuffled = shuffle;
        inputPath =  ipath; 
        labelPath = lpath;
        
        inputifs.open(inputPath, ios::binary);
        if (!inputifs.good()) { 
            cout << endl << "Input file does not exist: '" << ipath << "'" << endl;  
            throw runtime_error(""); //too lazy to write try and catch
        }
        inputifs.seekg(sizeof(int));
        inputifs.read((char*)&totalNumItems, sizeof(totalNumItems));
        inputifs.read((char*)&rows, sizeof(rows));
        inputifs.read((char*)&cols, sizeof(cols));
        totalNumItems = _byteswap_ulong(totalNumItems);
        rows = _byteswap_ulong(rows);
        cols = _byteswap_ulong(cols);
        inputSize = rows * cols;
        labelifs.open(labelPath, ios::binary);
        if (!inputifs.good()) {
            cout << endl << "Label file does not exist: '" << lpath << "'" << endl;
            throw runtime_error("");
        }
        
        inputData.reserve(totalNumItems);
        labelData.reserve(totalNumItems);
        inputifs.seekg(4 * sizeof(int));
        labelifs.seekg(2 * sizeof(int)); 
        //totalNumItems = 5000; //!FOR TESTING
        for (int i = 0; i < totalNumItems; i++) {
            Vectorf inputVec(inputSize);
            Vectorf labelVec(labelSize, lin::zeros);
            byte* inputByteArr = new byte[inputSize];
            byte label;
            inputifs.read((char*)inputByteArr, inputSize);
            labelifs.read((char*)&label, 1);
            labelVec[label] = 1;
            for (int j = 0; j < inputSize; j++) {
                inputVec[j] = inputByteArr[j];
            }
            labelData.push_back(labelVec);
            inputData.push_back(inputVec);
        }
        if (shuffled) { 
            vector<Vectorf> shuffledInputData(totalNumItems, Vectorf(inputSize));
            vector<Vectorf> shuffledLabelData(totalNumItems, Vectorf(labelSize));
            vector<int> shuffledNums(totalNumItems);
            for (int i = 0; i < totalNumItems; ++i) { shuffledNums[i] = i; }
            unsigned seed = chrono::system_clock::now().time_since_epoch().count();
            std::shuffle(shuffledNums.begin(), shuffledNums.end(), default_random_engine(seed));  
            for (int i = 0; i < totalNumItems; i++) { 
                shuffledInputData[i] = inputData[shuffledNums[i]];
                shuffledLabelData[i] = labelData[shuffledNums[i]];
            }
            inputData = shuffledInputData;
            labelData = shuffledLabelData;
        }
    }
    Batch next(int size) //get a new batch
    {
        if (curPos + size > totalNumItems) {
            size = totalNumItems - curPos;
            endReached = true;
        }
        Batch batch(size);
        for (int i = 0; i < size; i++) {
            InputLabelPair ilpair;
            ilpair.input = &inputData[curPos];
            ilpair.label = &labelData[curPos];
            batch[i] = ilpair;
            curPos++;
        }
        if (endReached) {
            curPos = 0;
            endReached = false;
        }
        return batch;
    }
};

namespace Func
{
    class AActFunction
    {
    public:
        virtual float forward(float x) = 0;
        virtual float backward(float x, float y) = 0;
        virtual Vectorf forward(Vectorf& vec)
        {
            Vectorf newVec(vec.size);
            for (size_t i = 0; i < vec.size; i++) {
                newVec[i] = forward(vec[i]);
            }
            return newVec;
        }
        virtual Vectorf backward(Vectorf& vec1, Vectorf& vec2)
        {
            Vectorf newVec(vec1.size);
            for (size_t i = 0; i < vec1.size; i++) {
                newVec[i] = backward(vec1[i], vec2[i]);
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
    class ALossFunction
    {
    public:
        virtual Vectorf forward(Vectorf& vec1, Vectorf& vec2) = 0;
        virtual Vectorf backward(Vectorf& vec1, Vectorf& vec2) = 0;
        virtual float numericLoss(Vectorf& vec1, Vectorf& vec2)
        {
            return forward(vec1, vec2).sum();
        }

        float operator () (Vectorf& vec1, Vectorf& vec2)
        {
            return numericLoss(vec1, vec2);
        }
    };

    namespace Act
    {
        class reLU : public AActFunction
        {
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
        };
        class lReLU : public AActFunction
        {
            float forward(float x) override
            {
                if (x > 0) return x;
                else return 0.01 * x;
            }
            float backward(float x, float y) override
            {
                if (x >= 0) return y;
                else return 0.01 * y;
            }
        };
        class stdLogistic : public AActFunction
        {
            float forward(float x) override
            {
                return 1 / (1 + exp(-x));
            }
            float backward(float x, float y) override
            {
                float ex = exp(x);
                return (ex / pow((ex + 1), 2)) * y;
            }
        };
        class softMax : public AActFunction
        {

        };
    }
    namespace Loss
    {
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
        class CrossEntropy : public ALossFunction
        {
            
        };
    }
    namespace WeightInit
    {
        Matrixf uniformInit(int rows, int cols)
        {
            Matrixf weightMat(rows, cols, lin::uniform, { -1, 1 });
            return weightMat; //matrix with uniform distribution from -1 to 1
        }
        Matrixf heInit(int rows, int cols)
        {
            float stddev = sqrt(2.0 / (double)cols);
            Matrixf weightMat(rows, cols, lin::normal, { 0, stddev });
            return weightMat; //matrix with He initialization: normal distribution with mean 0 and stddev = sqrt(2/(l-1))
        }
        Matrixf xavierInit(int rows, int cols)
        {
            float stddev = sqrt(2.0 / (((double)cols) + ((double)rows)));
            Matrixf weightMat(rows, cols, lin::normal, { 0, stddev });
            return weightMat; //matrix with Xavier initialization: normal distribution with mean 0 and stddev = sqrt(2/((l-1)+(l)))
        }
    }

    void showImg(Matrixf mat)
    {
        for (int i = 0; i < mat.rows; i++) {
            for (int j = 0; j < mat.cols; j++) {
                int x = mat[i][j];
                string c = "  ";
                if (x > 0) {
                    c = ": ";
                    if (x > 100) {
                        c = "o ";
                        if (x > 170) {
                            c = "# ";
                            if (x > 251) {
                                c = string(1, (char)254u) + " ";
                            }
                        }
                    }
                }
                cout << c;
            }
            cout << endl;
        }
    }
}
using namespace Func;

class ALayer
{
    //     #    <---- tumbleweed
};
class Linear : public ALayer
{
public:
    int inSize, outSize;
    Vectorf sums; //the weighted sums of each neuron
    Vectorf outs; //the outputs of each neuron
    Matrixf weights; //the layer's weight matrix
    Matrixf weightsGrad; //the gradient of the weight matrix
    Matrixf weightsGradSum; //the sum of multiple weight gradients (for SGD) 
    AActFunction& actFunc; //the activation function used by each neuron

    Vectorf* prevOuts; //the outputs of the previous layer
public:
    Linear(int inChan, int outChan, Func::AActFunction& actFnc, function<Matrixf(int, int)> weightInit = Func::WeightInit::heInit)
        : actFunc(actFnc)
    {
        inSize = inChan; 
        outSize = outChan;
        sums = Vectorf(outChan);
        outs = Vectorf(outChan);
        weights = weightInit(outChan, inChan);
        weightsGrad = Matrixf(outChan, inChan);
        weightsGradSum = Matrixf(outChan, inChan, lin::zeros);
    }

    Vectorf forward(Vectorf inVec)
    {
        sums = weights * inVec;
        outs = actFunc.forward(sums);
        return outs;
    }
    Vectorf backward(Vectorf sumGrad)
    {
        sumGrad = actFunc.backward(sums, sumGrad);
        weightsGrad = sumGrad.transposed() * (*prevOuts).asMatrix(); //?SLOW
        weightsGradSum += weightsGrad;
        sumGrad = weights.transposed() * sumGrad;
        return sumGrad;
    }
    void update(float lr)
    {
        Matrixf updateMat = weightsGradSum * lr;
        weights -= updateMat;
    }
};
class Network
{
public:
    vector<Linear> layers;
    ALossFunction& lossFunc;
    Vectorf* output;
public:
    Network(initializer_list<Linear> lrs, ALossFunction& lossFnc) : lossFunc(lossFnc)
    {
        for (auto it = lrs.begin(); it != lrs.end(); it++) {
            Linear* lr = new Linear(it->inSize, it->outSize, it->actFunc);
            layers.push_back(*lr);
        }
        for (auto it = layers.begin() + 1; it != layers.end(); it++) {
            it->prevOuts = &((it - 1)->outs);
        }
        output = &(layers[layers.size() - 1].outs);
    }
    
    void forward(Vectorf inputVec)
    {
        layers[0].prevOuts = new Vectorf(inputVec.nums); //?SLOW
        for (int i = 0; i < layers.size(); i++) {
            inputVec = layers[i].forward(inputVec);
        }
    }
    void backward(Vectorf label)
    {
        Vectorf sumGrad = lossFunc.backward(*output, label);
        for (int i = layers.size() - 1; i >= 0; i--) {
            sumGrad = layers[i].backward(sumGrad);
        }
    }
    void optimize(float lr) //optimize with stochastic gradient descent 
    {
        for (int i = 0; i < layers.size(); i++) {

            layers[i].update(lr);
        }
    }
    void zeroGrad()
    {
        for (int i = 0; i < layers.size(); i++) {
            layers[i].weightsGradSum = layers[i].weightsGradSum * 0;
        }
    }
};

#include <locale>
#include <codecvt>
string ExePath()
{
    TCHAR buffer[MAX_PATH] = { 0 };
    GetModuleFileName(NULL, buffer, MAX_PATH);
    wstring str;
    wstring::size_type pos = wstring(buffer).find_last_of(L"\\/");
    str = wstring(buffer).substr(0, pos);
    pos = wstring(str).find_last_of(L"\\/");
    str = wstring(str).substr(0, pos);
    using convert_type = std::codecvt_utf8<wchar_t>;
    wstring_convert<convert_type, wchar_t> converter;
    return converter.to_bytes(str);
}

string DATA_DIR = R"()"; //If the files aren't found, enter the path where the MNIST ubyte files are located in the parentheses.

//int main()
//{
//    if (DATA_DIR == "") DATA_DIR = ExePath() + R"(\data_mnist\)";
//    else if (DATA_DIR.back() != '\\') DATA_DIR += "\\";
//    DataLoader trainLoader(DATA_DIR + "train-images.idx3-ubyte", DATA_DIR + "train-labels.idx1-ubyte");
//    DataLoader testLoader(DATA_DIR + "t10k-images.idx3-ubyte", DATA_DIR + "t10k-labels.idx1-ubyte");
//
//    Network nn({ Linear(784, 10, *(new Act::lReLU())),
//                 Linear(10, 10, *(new Act::lReLU())),
//                 Linear(10, 10, *(new Act::lReLU())) }, *(new Loss::MSE()));
//
//    size_t batchSize = 8;
//    float learningRate = 0.00001;
//    while(cin.ignore())
//    {
//        Batch bat = trainLoader.next(batchSize);
//        float batAvgLoss = 0;
//        
//        nn.zeroGrad();
//        for (size_t i = 0; i < bat.size(); i++) {
//            nn.forward(*bat[i].input);
//            batAvgLoss += nn.lossFunc(*nn.output, *bat[i].label);
//            nn.backward(*bat[i].label);
//        }
//        nn.optimize(learningRate);
//
//        cout << "Average loss: " << batAvgLoss / bat.size();
//    }
//}