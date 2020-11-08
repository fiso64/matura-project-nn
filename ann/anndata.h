#pragma once
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

using namespace std; //this is good practice
namespace lin = linalg;

namespace anndata
{
	struct InputLabelPair;
	typedef lin::Vector<float> Vectorf;
	typedef lin::Matrix<float> Matrixf;
	typedef vector<InputLabelPair> Batch;

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
            inputPath = ipath;
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
}
