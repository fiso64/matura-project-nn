#include <iostream>
#include <locale>
#include <codecvt>
#include <thread>
#include <iomanip>

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
        void lap(std::string str = "", bool print = true) {
            stop(str, print);
            t1 = std::chrono::high_resolution_clock::now();
        }
        void stop(std::string str = "", bool print = true) {
            t2 = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            if (print)
                std::cout << "\nTIME ELAPSED: " << duration << " microseconds = "
                << duration / 1000000.0 << " seconds. " << str;
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
    //TODO: CrossEntropyLoss & softMax & Momentum
    //TODO: Solve the Network and SGD template mess
    //TODO: Calling forward & backward on batches
    //TODO: Network with different kinds of layers
}
namespace lin = linalg;
using namespace stuff;
using namespace std;



//##############################################################################################################################################
std::string DATA_DIR = R"()"; //If the files aren't found, enter the path where the train set and test set files are located in the parentheses. #
//##############################################################################################################################################

void trainLoop(nnet::Network<nnet::Linear>& net, optim::SGD<nnet::Linear>& optimizer, data::DataLoader& trainLoader, int verbose = 1)
{
    int i = 0;
    float avgLoss = 0;
    int numRight = 0;
    int printSpeed = 5000;
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
                std::cout << "learning rate: " << optimizer.learningRate << endl;
                std::cout << "Average loss over " << amount << " batches (size=" << bat.size() << "): " << avgLoss << endl;
                std::cout << "Correctly predicted of " << bat.size() * amount 
                    << " items: " << (float)(numRight * 100) / (bat.size() * amount) << "%\n\n";
            }
            avgLoss = 0;
            numRight = 0;
        }
    }
}
void testNet(nnet::Network<nnet::Linear>& nn, data::DataLoader& testLoader)
{
    float avgLoss = 0;
    int numRight = 0;
    int predDigit, labelDigit;
    data::Batch bat = testLoader.all();

    for (size_t i = 0; i < bat.size(); i++) {
        nn.forward(*bat[i].input);
        avgLoss += nn.lossFunc(*nn.output, *bat[i].label);
        predDigit = distance((*nn.output).nums.begin(), max_element((*nn.output).nums.begin(), (*nn.output).nums.end()));
        labelDigit = distance((*bat[i].label).nums.begin(), max_element((*bat[i].label).nums.begin(), (*bat[i].label).nums.end()));
        if (predDigit == labelDigit) { numRight++; }
    }

    std::cout << "\nAverage loss over " << bat.size() << " items: " << avgLoss/bat.size()  << endl;
    std::cout << "Correctly predicted " << (float)(numRight * 100) / bat.size() << "%\n\n";
}
int main()
{
    DATA_DIR = getDataPath(DATA_DIR);

    data::MNIST trainSet(DATA_DIR + "train-images.idx3-ubyte", DATA_DIR + "train-labels.idx1-ubyte");
    data::MNIST testSet(DATA_DIR + "t10k-images.idx3-ubyte", DATA_DIR + "t10k-labels.idx1-ubyte");
    data::DataLoader trainLoader(trainSet, 64);
    data::DataLoader testLoader(testSet);

    auto weightInitFunc = func::weightInit::heInitHalfStd;
    func::loss::MSE lossFunc;

    nnet::Network<nnet::Linear> net ({ 
        new nnet::Linear(784,16, (new func::act::reLU())),
        new nnet::Linear(16, 16, (new func::act::reLU())),
        new nnet::Linear(16, 16, (new func::act::reLU())),
        new nnet::Linear(16, 16, (new func::act::reLU())),
        new nnet::Linear(16, 10, (new func::act::stdLogisticLinearEnds())) }, 
        lossFunc, weightInitFunc);

    optim::SGD<nnet::Linear> optimizer(net.layers, 0.1, 0.001);

    int epoch = 0;
    while (true) {
        epoch++;
        trainLoop(net, optimizer, trainLoader, 1);
        cout << "Epoch " << epoch << " complete." << endl;
        while (true) {
            cout << "Press enter to train again. 0 to test model on testset. ";
            string x;
            getline(cin, x);
            if (x == "0") testNet(net, testLoader);
            else break;
        }
    }
}