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

namespace stuff
{
    class Timer
    {
    public:
        std::chrono::steady_clock::time_point t1;
        std::chrono::steady_clock::time_point t2;
        long long duration = 0;
    public:
        Timer() {
            t1 = std::chrono::high_resolution_clock::now();
        }
        void start() {
            t1 = std::chrono::high_resolution_clock::now();
        }
        void lap(std::string str = "") {
            stop(str);
            t1 = std::chrono::high_resolution_clock::now();
        }
        void stop(std::string str = "") {
            t2 = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            std::cout << "\nTIME ELAPSED: " << duration << " microseconds = "
                << duration / 1000000.0 << " seconds. " << str;
        }
        void stop(bool print = false) {
            t2 = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            if (print) {
                std::cout << "\nTIME ELAPSED: " << duration << " microseconds = "
                    << duration / 1000000.0 << " seconds. ";
            }
        }
        long long microSeconds() {
            stop(false);
            return duration;
        }
        float milliSeconds() {
            stop(false);
            return duration / 1000.0;
        }
        float seconds() {
            stop(false);
            return duration / 1000000.0;
        }
        float minutes() {
            stop(false);
            return duration / 60000000.0;
        }
    };
    std::string getExePath()
    {
        TCHAR buffer[MAX_PATH] = { 0 };
        GetModuleFileName(NULL, buffer, MAX_PATH);
        std::wstring str;
        std::wstring::size_type pos = std::wstring(buffer).find_last_of(L"\\/");
        str = std::wstring(buffer).substr(0, pos);
        pos = std::wstring(str).find_last_of(L"\\/");
        str = std::wstring(str).substr(0, pos);
        using convert_type = std::codecvt_utf8<wchar_t>;
        std::wstring_convert<convert_type, wchar_t> converter;
        return converter.to_bytes(str);
    }
    std::string getDataPath(std::string dir)
    {
        if (dir == "") dir = getExePath() + R"(\data_mnist\)";
        else if (dir.back() != '\\') dir += "\\";
        return dir;
    }
    unsigned char* readBMP(const char* filename)
    {
        int i;
        FILE** f = new FILE*;
        fopen_s(f, filename, "rb");
        unsigned char info[54];
        fread(info, sizeof(unsigned char), 54, *f);
        int width = *(int*)&info[18];
        int height = *(int*)&info[22];
        int size = 3 * width * height;
        unsigned char* data = new unsigned char[size];
        fread(data, sizeof(unsigned char), size, *f);
        fclose(*f);

        for (i = 0; i < size; i += 3) {
            unsigned char tmp = data[i];
            data[i] = data[i + 2];
            data[i + 2] = tmp;
        }

        return data;
    }
    template <class T>
    int indexOfMax(linalg::Vector<T>& vec)
    {
        return std::distance(vec.nums.begin(), std::max_element(vec.nums.begin(), vec.nums.end()));
    }
    bool isFloat(std::string myString) {
        std::istringstream iss(myString);
        float f;
        iss >> std::noskipws >> f;
        return iss.eof() && !iss.fail();
    }
    template<class T, class... Args>
    void print(T t)
    {
        std::cout << t;
        std::cout << std::endl;
    }
    template<class T, class... Args>
    void print(T t, Args... args)
    {
        std::cout << t;
        print(args...);
    }
    //TODO: test on custom images
    //TODO: Make SGD an actual optimizer
    //TODO: CrossEntropyLoss & softMax & Momentum
    //TODO: Solve the Network and SGD template mess
    //TODO: Calling forward & backward on batches
    //TODO: Network with different kinds of layers
    //TODO: Conv layers
    //TODO: Autograd
    //TODO: Maybe make this work in a python module
}
namespace lin = linalg;
using namespace stuff;
using namespace std;

/*some tests
###
batch size: 64
loss function: MSE
net: Linear(500,relu), Linear(150,relu), Linear(10,stdLogistic)
optimizer: lr=0.1, decay=0.01

error rate 5% after 1 epoch
2.7% after 3 more epochs & by manually adjusting learn rate
###
*/

//##############################################################################################################################################
string DATA_DIR = R"()"; //If the files aren't found, enter the path where the train set and test set files are located in the parentheses. #
//##############################################################################################################################################

void trainLoop(nnet::Network&, optim::SGD&, data::DataLoader&, int);
void testNet(nnet::Network&, data::DataLoader&);
void testNet(nnet::Network&, lin::Vector<float>&, bool);

int main()
{
    //auto img = readBMP(R"(D:\Users\sokfi\Downloads\Untitled-1.bmp)");
    //for (size_t i = 0; i < 28 * 3; i += 3) {
    //    for (size_t j = 0; j < 28 * 3; j += 3) {
    //        //int tmp1 = img[i * 28 + j];
    //        //int tmp2 = img[i * 28 + j + 1];
    //        //int tmp3 = img[i * 28 + j + 2];
    //        cout << (int)img[i * 28 + j] << ", ";
    //    }
    //    cout << endl;
    //}


    DATA_DIR = getDataPath(DATA_DIR);
    string trainImagesPath = "path";
    //create datasets from the mnist files
    data::MNIST trainSet(DATA_DIR + "train-images.idx3-ubyte", DATA_DIR + "train-labels.idx1-ubyte");
    data::MNIST testSet(DATA_DIR + "t10k-images.idx3-ubyte", DATA_DIR + "t10k-labels.idx1-ubyte");
    //create a data loaders from the datasets
    data::DataLoader trainLoader(trainSet, 64);
    data::DataLoader testLoader(testSet);

    //create network taking 784 inputs, passing them through 5 layers and returning 10 outputs
    nnet::Network net({ 
        new nnet::Linear(784,16, func::act::reLU()),
        new nnet::Linear(16, 16, func::act::reLU()),
        new nnet::Linear(16, 16, func::act::reLU()),
        new nnet::Linear(16, 16, func::act::reLU()),
        new nnet::Linear(16, 10, func::act::stdLogistic()) },
        new func::loss::MSE()); //use mean squared error as loss function

    //create stochastic gradient descent optimizer with learn rate=0.1 and learn rate decay speed=0.001
    optim::SGD optimizer(net.layers, 0.1, 0.01);
    
    int epoch = 0;
    cout << "Start train" << endl; 
    while (true) {
        epoch++;
        Timer t;
        trainLoop(net, optimizer, trainLoader, 1);
        cout << "Epoch " << epoch << " complete in " << t.seconds() << " seconds." << endl;
        while (true) {
            cout << std::setprecision(5);
            cout << "\nEnter: train again\n0:     test model on testset\n"
                << "000:   exit\nCurrent learn rate: " << optimizer.learnRate << ", enter a number to change.\n";
            string x;
            getline(cin, x);
            if (x == "0") testNet(net, testLoader);
            else if (x == "000") goto endLoop;
            else if (x == "") break;
            else if (isFloat(x)) { optimizer.learnRate = std::stod(x); break; }
        }
    }

endLoop:
    while (true) {
        auto img = *testLoader.next(1)[0].input;
        testNet(net, img, true);
        cin.ignore();
    }
}

void trainLoop(nnet::Network& net, optim::SGD& optimizer, data::DataLoader& trainLoader, int verbose = 1)
{
    int i = 0;
    float avgLoss = 0;
    int numRight = 0;
    int printSpeed = 5000;
    trainLoader.reset();
    while (!trainLoader.endReached)
    {
        float batAvgLoss = 0;
        int predDigit, labelDigit;
        data::Batch bat = trainLoader.next();

        optimizer.zeroGrad();
        for (size_t i = 0; i < bat.size(); i++)
        {
            net.forward(*bat[i].input);
            batAvgLoss += net.lossFunc(*net.output, *bat[i].label);
            predDigit = distance((*net.output).nums.begin(), max_element((*net.output).nums.begin(), (*net.output).nums.end()));
            labelDigit = distance((*bat[i].label).nums.begin(), max_element((*bat[i].label).nums.begin(), (*bat[i].label).nums.end()));
            if (predDigit == labelDigit) numRight++;
            net.backward(*bat[i].label);
        }
        optimizer.step();
        
        avgLoss += batAvgLoss / bat.size();
        int amount = ((10000 - printSpeed) / bat.size());
        if (verbose > 0 && ++i % amount == 0) {
            avgLoss /= amount;
            if (verbose == 1) {
                std::cout << std::fixed;
                std::cout << std::setprecision(5);
                std::cout << "Average loss over " << bat.size() * amount << " items: " << avgLoss;
                std::cout << std::setprecision(2);
                std::cout << ", correctly predicted " << (float)(numRight * 100) / (bat.size() * amount) << "%\n";
            }
            else if (verbose == 2) {
                std::cout << std::fixed;
                std::cout << std::setprecision(3);
                net.output->print("latest output: ", "\n", false);
                bat[bat.size() - 1].label->print("latest label : ", "\n", false);
                std::cout << std::setprecision(6);
                std::cout << "learning rate: " << optimizer.learnRate << endl;
                std::cout << "Average loss over " << amount << " batches (size=" << bat.size() << "): " << avgLoss << endl;
                std::cout << "Correctly predicted of " << bat.size() * amount
                    << " items: " << (float)(numRight * 100) / (bat.size() * amount) << "%\n\n";
            }
            avgLoss = 0;
            numRight = 0;
        }
    }
}

void testNet(nnet::Network& net, data::DataLoader& testLoader)
{
    data::Batch bat = testLoader.all();

    float avgLoss = 0;
    int numRight = 0;
    for (size_t i = 0; i < bat.size(); i++) {
        net.forward(*bat[i].input);
        avgLoss += net.lossFunc(*net.output, *bat[i].label);
        int predDigit = indexOfMax(*net.output);
        int labelDigit = indexOfMax(*bat[i].label);
        if (predDigit == labelDigit) { numRight++; }
    }

    cout << "\nAverage loss over " << bat.size() << " items: " << avgLoss / bat.size() << endl;
    cout << "Correctly predicted " << (numRight * 100.0) / bat.size() << "%\n\n";
}

void testNet(nnet::Network& net, lin::Vector<float>& input, bool showImg)
{
    if (showImg) data::MNIST::showImg(input);
    net.forward(input);
    net.output->print("","",false);
    cout << "\n Looks like a " <<
        distance((*net.output).nums.begin(), max_element((*net.output).nums.begin(), (*net.output).nums.end())) << endl;
}