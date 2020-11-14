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
    
    //A std::vector of inputs with a corresponding std::vector of labels
    struct InputLabelPair
    {
        Vectorf* input;
        Vectorf* label;
    };

    class IDataSet
    {
    public:
        int totalNumItems; //number of items in the dataset
        int inputSize; //size of the input std::vector
        int labelSize; //size of the label std::vector
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
        MNIST(std::string ipath, std::string lpath, bool verbose = true)
        {
            inputPath = ipath;
            labelPath = lpath;
            loadData();
            if (verbose) std::cout << totalNumItems << " items loaded from '" << ipath << "'\n";
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
            std::vector<Vectorf> shuffledInputData(totalNumItems, Vectorf(inputSize));
            std::vector<Vectorf> shuffledLabelData(totalNumItems, Vectorf(labelSize));
            std::vector<int> shuffledNums(totalNumItems);
            for (int i = 0; i < totalNumItems; ++i) { shuffledNums[i] = i; }
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::shuffle(shuffledNums.begin(), shuffledNums.end(), std::default_random_engine(seed));
            for (int i = 0; i < totalNumItems; i++) {
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
        //Write from the ubyte mnist files to the input and label std::vectors.
        void loadData() 
        {
            inputifs.open(inputPath, std::ios::binary);
            if (!inputifs.good()) {
                std::cout << "\nInput file does not exist: '" << inputPath << "'\n";
                throw std::runtime_error(""); //too lazy to write try and catch
            }
            int rows, cols;
            inputifs.seekg(sizeof(int));
            inputifs.read((char*)&totalNumItems, sizeof(totalNumItems));
            inputifs.read((char*)&rows, sizeof(rows));
            inputifs.read((char*)&cols, sizeof(cols));
            rows = _byteswap_ulong(rows);
            cols = _byteswap_ulong(cols);

            //###################################
            totalNumItems = _byteswap_ulong(totalNumItems);
            inputSize = rows * cols;
            labelSize = 10;
            //###################################

            labelifs.open(labelPath, std::ios::binary);
            if (!inputifs.good()) {
                std::cout << "\nLabel file does not exist: '" << labelPath << "'\n";
                throw std::runtime_error("");
            }

            inputData.reserve(totalNumItems);
            labelData.reserve(totalNumItems);
            inputifs.seekg(4 * sizeof(int));
            labelifs.seekg(2 * sizeof(int));
            for (int i = 0; i < totalNumItems; i++) {
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
    };

    //Holds a ADataSet object and is resposible for getting batches from it.
    class DataLoader
    {
    public:
        int curPos = 0;
        int batchSize = 1;
        bool endReached = false;
        bool restartAfterEndReached;
        bool shuffled;
        IDataSet& dataSet;
    public:
        DataLoader(IDataSet& datSet, int batSize = 1, bool shuffle = true, bool loop = true) : dataSet(datSet)
        {
            batchSize = batSize;
            shuffled = shuffle;
            restartAfterEndReached = loop;
            if (shuffle) dataSet.shuffle();
        }
        //Get a batch of a size.
        //IN: size of the batch to return
        //OUT: a batch of size 'size'
        Batch next(int batSize = -1)
        {
            if (batSize == -1) batSize = batchSize;
            if (!endReached && curPos + batSize >= dataSet.totalNumItems) {
                batSize = dataSet.totalNumItems - curPos;
                endReached = true;
            }
            else if (endReached) { endReached = false; }

            Batch batch(batSize);
            for (size_t i = 0; i < batSize; i++) {
                batch[i] = dataSet.getItem(curPos);
                curPos++;
            }

            if (endReached && restartAfterEndReached) { curPos = 0; }
            return batch;
        }
        //Return all items from the dataset in one batch.
        //OUT: batch of size dataSet.totalNumItems containing all items of the dataset.
        Batch all()
        {
            curPos = 0;
            Batch batch(dataSet.totalNumItems);
            for (size_t i = 0; i < dataSet.totalNumItems; i++) {
                batch[i] = dataSet.getItem(curPos);
                curPos++;
            }
            if (restartAfterEndReached) curPos = 0;
            return batch;
        }
    };
}
