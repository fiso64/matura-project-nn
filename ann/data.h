#pragma once
#include <stdarg.h>
#include <random>
#include <map>
#include <fstream>
#include <iterator>
#include <intrin.h>
#include <chrono>
#include <stdexcept>
#include <cassert>

#include "linalg.h"

namespace data
{
	struct InputLabelPair;
	typedef linalg::Vector<float> Vectorf;
	typedef linalg::Matrix<float> Matrixf;
	typedef std::vector<InputLabelPair> Batch;
    
    //A vector of inputs with a corresponding vector of labels
    struct InputLabelPair
    {
        Vectorf* input;
        Vectorf* label;
        Vectorf a;
    };

    class IDataSet
    {
    public:
        int size; //number of items in the dataset
        int inputSize; //size of the input vectors
        int labelSize; //size of the label vectors
    public:
        //Get a single InputLabelPair from the data.
        //IN: index
        virtual InputLabelPair getItem(int ind) = 0;

        virtual void shuffle() = 0;
    };

    class MNIST : public IDataSet
    {
    public:
        std::ifstream inputifs;
        std::ifstream labelifs;
        std::string inputPath;
        std::string labelPath;
        std::vector<Vectorf> inputData;
        std::vector<Vectorf> labelData;
    public:
        //IN: input file path, label file path
        MNIST(std::string ipath, std::string lpath)
        {
            inputPath = ipath;
            labelPath = lpath;
            loadData();
            std::cout << size << " items loaded from '" << ipath << "'\n";
        }
        //Creates a MNIST dataset from the default directory if files are present
        //IN: "train" or "test"
        MNIST(std::string type)
        {
            std::string dataDir = getDefaultDataPath();
            if (type == "train") {
                inputPath = dataDir + "train-images.idx3-ubyte";
                labelPath = dataDir + "train-labels.idx1-ubyte";
            }
            else if (type == "test") {
                inputPath = dataDir + "t10k-images.idx3-ubyte";
                labelPath = dataDir + "t10k-labels.idx1-ubyte";
            }
            else {
                std::cout << "Invalid MNIST constructor argument. Only \"train\" or \"test\".";
            }

            loadData();
            std::cout << size << " items loaded from '" << inputPath << "'\n";
        }

        InputLabelPair getItem(int ind) override
        {
            InputLabelPair ilpair;
            ilpair.input = &inputData[ind];
            ilpair.label = &labelData[ind];
            return ilpair;
        }

        void shuffle() override
        {
            std::vector<Vectorf> shuffledInputData(size, Vectorf(inputSize));
            std::vector<Vectorf> shuffledLabelData(size, Vectorf(labelSize));
            std::vector<int> shuffledNums(size);
            for (int i = 0; i < size; ++i) { shuffledNums[i] = i; }
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::shuffle(shuffledNums.begin(), shuffledNums.end(), std::default_random_engine(seed));
            for (int i = 0; i < size; i++) {
                shuffledInputData[i] = inputData[shuffledNums[i]];
                shuffledLabelData[i] = labelData[shuffledNums[i]];
            }
            inputData = shuffledInputData;
            labelData = shuffledLabelData;
        }

        static void showImg(Vectorf& image)
        {
            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    int x = image[i * 28 + j];
                    std::string c = "  ";
                    if (x > 0) {
                        c = ": ";
                        if (x > 100) {
                            c = "o ";
                            if (x > 170) {
                                c = "# ";
                                if (x > 251) {
                                    c = std::string(1, (char)254u) + " ";
                                }
                            }
                        }
                    }
                    std::cout << c;
                }
                std::cout << std::endl;
            }
        }
    protected:
        //Write from the ubyte mnist files to the input and label vectors.
        void loadData() 
        {
            inputifs.open(inputPath, std::ios::binary);
            if (!inputifs.good()) {
                std::cout << "\nInput file not found: '" << inputPath << "'\n";
                throw std::runtime_error(""); //too lazy to write try and catch
            }
            int rows, cols;
            inputifs.seekg(sizeof(int));
            inputifs.read((char*)&size, sizeof(size));
            inputifs.read((char*)&rows, sizeof(rows));
            inputifs.read((char*)&cols, sizeof(cols));
            rows = _byteswap_ulong(rows);
            cols = _byteswap_ulong(cols);

            //###################################
            size = _byteswap_ulong(size);
            inputSize = rows * cols;
            labelSize = 10;
            //###################################

            labelifs.open(labelPath, std::ios::binary);
            if (!labelifs.good()) {
                std::cout << "\nLabel file not found: '" << labelPath << "'\n";
                throw std::runtime_error("");
            }

            inputData.reserve(size);
            labelData.reserve(size);
            inputifs.seekg(4 * sizeof(int));
            labelifs.seekg(2 * sizeof(int));
            for (int i = 0; i < size; i++) {
                Vectorf inputVec(inputSize);
                Vectorf labelVec(labelSize, linalg::zeros);
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
        }

        std::string getDefaultDataPath()
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
            std::string dir = converter.to_bytes(str) + R"(\data_mnist\)";
            return dir;
        }
    };

    //Holds a ADataSet object and is resposible for getting batches from it.
    class DataLoader
    {
    protected:
        bool _endReached = false;
        bool _restart = false;
    public:
        int curPos = 0;
        int batchSize = 1;
        bool restartAfterEndReached;
        bool shuffled;
        IDataSet& dataSet;
    public:
        DataLoader(IDataSet& datSet, int batSize = 1, bool shuffle = true, bool autoLoop = true) : dataSet(datSet)
        {
            batchSize = batSize;
            shuffled = shuffle;
            restartAfterEndReached = autoLoop;
            if (shuffle) dataSet.shuffle();
        }

        //checks if end has been reached, resets to false if called a second time and if restartAfterEndReached=true
        //for use in loops
        bool endReached()
        {
            _restart = restartAfterEndReached && !_restart && _endReached;
            if (_restart) {
                curPos = 0;
                _endReached = false;
                _restart = false;
                return true;
            }
            else {
                return _endReached;
            }
        }

        //Get a batch of InputLabelPairs.
        //IN: size of the batch to return
        //OUT: a batch of InputLabelPairs of the specified size
        Batch next(int batSize = -1)
        {
            if (batSize == -1) batSize = batchSize;
            if (curPos + batSize >= dataSet.size) {
                batSize = dataSet.size - curPos;
                _endReached = true;
            }

            Batch batch(batSize);
            for (size_t i = 0; i < batSize; i++) {
                batch[i] = dataSet.getItem(curPos);
                curPos++;
            }

            if (_endReached && restartAfterEndReached) curPos = 0;
            return batch;
        }

        //Return all items from the dataset in one batch.
        //OUT: batch of size dataSet.totalNumItems containing all items of the dataset.
        Batch all()
        {
            curPos = 0;
            Batch batch(dataSet.size);
            for (size_t i = 0; i < dataSet.size; i++) {
                batch[i] = dataSet.getItem(curPos);
                curPos++;
            }
            if (restartAfterEndReached) {
                curPos = 0;
                _endReached = false;
            }
            else {
                _endReached = true;
            }
            return batch;
        }

        void reset()
        {
            curPos = 0;
            _endReached = false;
        }
    };
}
