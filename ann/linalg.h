#pragma once
#include <iostream>
#include <string>
#include <algorithm>
#include <cmath>
#include <vector>
#include <typeinfo>
#include <functional>
#include <random>
#include <stdarg.h>
#include <stdlib.h>

//If you're reading this, turn back while you can

//...

//I am so sorry

namespace linalg
{
    using namespace std;
    enum initType { zeros, ones, identity, uniform, normal };
    template <class T>
    class Matrix;
    template <class T>
    class VectorTest
    {
    public:
        vector<T> nums;
        int size;

        VectorTest(int s, T* varr = NULL)
        {
            size = s;
            if (varr != NULL) {
                /*for (size_t i = 0; i < size; i++) {
                    nums[i] = varr[i];
                }*/
                nums.assign(varr, varr + size);
            }
            else nums.reserve(s);
        }
        VectorTest(int s, initType type)
        {
            size = s;
            nums.reserve(s);
            switch (type)
            {
            case linalg::zeros:
                for (int i = 0; i < size; i++) { nums.push_back(0); }
                break;
            default:
                break;
            }
        }
        T& operator [] (int i) { return nums[i]; }
    };
    template <class T>
    class Vector
    {
    public:
        vector<T> nums;
        size_t size;

        Vector()
        {
            size = 0;
            nums = vector<T>();
        }
        Vector(int s, T* varr = NULL)
        {
            size = s;
            if (varr != NULL) { nums.assign(varr, varr + size); }
            else nums = vector<T>(size);
        }
        Vector(vector<T> varr)
        {
            size = varr.size();
            nums = varr;
        }
        Vector(initializer_list<T> initNums)
        {
            size = distance(initNums.begin(), initNums.end());
            nums = vector<T>(size);
            int i = 0;
            for (T num : initNums) { nums[i] = num; i++; }
        }
        Vector(int s, initType type, initializer_list<T> args = {})
        {
            size = s;
            nums = vector<T>(size);
        retryInitType:
            switch (type)
            {
            case linalg::zeros:
                for (int i = 0; i < size; i++) { nums[i] = 0; }
                break;
            case linalg::ones:
                for (int i = 0; i < size; i++) { nums[i] = 1; }
                break;
            case linalg::uniform:
            {
                if (typeid(T) == typeid(float) || typeid(T) == typeid(double) || typeid(T) == typeid(float))
                {
                    T min = -1, max = 1;
                    if (args.size() > 0) {
                        min = *(args.begin());
                        if (args.size() > 1) { max = *(args.end() - 1); }
                    }
                    random_device dev;
                    default_random_engine gen(dev());
                    uniform_real_distribution<T> dis(min, max);
                    for (int i = 0; i < size; i++) { nums[i] = dis(gen); }
                }
                break;
            }
            case linalg::normal:
            {
                if (typeid(T) == typeid(float) || typeid(T) == typeid(double) || typeid(T) == typeid(float))
                {
                    T mean = 0, stddev = 1;
                    if (args.size() > 0) {
                        mean = *(args.begin());
                        if (args.size() > 1) { stddev = *(args.end() - 1); }
                    }
                    random_device dev;
                    default_random_engine gen(dev());
                    normal_distribution<T> dis(mean, stddev);
                    for (int i = 0; i < size; i++) { nums[i] = dis(gen); }
                }
                break;
            }
            case linalg::identity:
                type = linalg::zeros;
                goto retryInitType;
                break;
            default:
                break;
            }
        }
        /*Vector(const Vector& vec)
        {
            size = vec.size;
            nums = make_shared<vector<T>>(size);
            for (int i = 0; int i < vec.size; int i++) {
                nums[i] = (*vec.nums)[i]
            }
        }
        */
        //Vector(Vector&& vec)
        //{
        //    size = vec.size;
        //    nums = vec.nums;
        //    vec.size = 0;
        //    //vec.nums = nullptr;
        //}

        void print(string str = "", string str2 = "", bool newLines = true)
        {
            if (newLines) cout << endl;
            cout << str << " [" << nums[0];
            for (int i = 1; i < size; i++) { cout << ", " << nums[i]; }
            cout << "]";
            cout << str2;
            if (newLines) cout << endl;
        }
        T dot(Vector vec)
        {
            T res = 0;
            for (int i = 0; i < size; i++) { res += nums[i] * vec.nums[i]; }
            return res;
        }
        //********************************************************************************************************
        //*transposes matrix
        //*IN: none
        //*OUT: matrix with transposed values
        //*PRE: non-empty vector
        //*POST: new object on stack, move
        //*TEST: B 1,2,3,4    W 1,2
        //********************************************************************************************************
        Matrix<T> transposed()
        {
            Matrix<T> newMat(size, 1, nums);
            return newMat;
        }
        float magnitude()
        {
            float res = 0;
            for (int i = 0; i < size; i++) { res += nums[i] * nums[i]; }
            return sqrt(res);
        }
        T sum()
        {
            T res = 0;
            for (int i = 0; i < size; i++) {
                res += nums[i];
            }
            return res;
        }

        Matrix<T> asMatrix(int m = 1, int n = -1)
        {
            if (n == -1) n = size;
            Matrix<T> newMat(m, n, nums);
            return newMat;
        }
        bool operator == (Vector vec)
        {
            for (int i = 0; i < size; i++) { 
                if (nums[i] != vec.nums[i]) return false; 
            }
            return true;
        }
        T& operator [] (int i) { return nums[i]; }
        Vector operator + (Vector vec)
        {
            Vector newVec(size);
            for (int i = 0; i < size; i++) { newVec.nums[i] = nums[i] + vec.nums[i]; }
            return newVec;
        }
        Vector operator + (T a)
        {
            Vector newVec(size);
            for (int i = 0; i < size; i++) { newVec.nums[i] = nums[i] + a; }
            return newVec;
        }
        Vector operator - (Vector vec)
        {
            Vector newVec(size);
            for (int i = 0; i < size; i++) { newVec.nums[i] = nums[i] - vec.nums[i]; }
            return newVec;
        }
        Vector operator - (T a)
        {
            Vector newVec(size);
            for (int i = 0; i < size; i++) { newVec.nums[i] = nums[i] - a; }
            return newVec;
        }
        T operator * (Vector vec)
        {
            T res = 0;
            for (int i = 0; i < size; i++) { res += nums[i] * vec.nums[i]; }
            return res;
        }
        Vector operator * (T a)
        {
            Vector newVec(size);
            for (int i = 0; i < size; i++) { newVec.nums[i] = nums[i] * a; }
            return newVec;
        }
        Vector operator / (T a)
        {
            Vector newVec(size);
            for (int i = 0; i < size; i++) { newVec.nums[i] = nums[i] / a; }
            return newVec;
        }
    };
    template <class T>
    class Matrix
    {
    public:
        vector<T> nums;
        int rows;
        int cols;
        int size;
        
        Matrix()
        {
            rows = 0;
            cols = 0;
            size = 0;
            nums = vector<T>();
        }
        Matrix(int m, int n, T* marr = NULL)
        {
            rows = m;
            cols = n;
            size = m * n;
            if (marr != NULL) { nums.assign(marr, marr + size); }
            else nums = vector<T>(size);
        }
        Matrix(int m, int n, vector<T> marr)
        {
            rows = m;
            cols = n;
            size = m * n;
            nums = marr;
        }
        Matrix(int m, int n, initializer_list<T> initNums)
        {
            rows = m;
            cols = n;
            size = m * n;
            nums = vector<T>(size);
            int i = 0;
            for (T num : initNums) { nums[i] = num; i++; }
        }
        Matrix(int m, int n, initType type, initializer_list<T> args = {})
        {
            rows = m;
            cols = n;
            size = m * n;
            nums = vector<T>(size);
            switch (type)
            {
            case linalg::zeros:
                for (int i = 0; i < size; i++) { nums[i] = 0; }
                break;
            case linalg::ones:
                for (int i = 0; i < size; i++) { nums[i] = 1; }
                break;
            case linalg::uniform:
            {
                if (typeid(T) == typeid(float) || typeid(T) == typeid(double) || typeid(T) == typeid(float))
                {
                    T min = -1, max = 1;
                    if (args.size() > 0) { min = *(args.begin());
                        if (args.size() > 1) { max = *(args.end() - 1); } }
                    random_device dev;
                    default_random_engine gen(dev());
                    uniform_real_distribution<T> dis(min, max);
                    for (int i = 0; i < size; i++) { nums[i] = dis(gen); }
                }
                break;
            }
            case linalg::normal:
            {
                if (typeid(T) == typeid(float) || typeid(T) == typeid(double) || typeid(T) == typeid(float))
                {
                    T mean = 0, stddev = 1;
                    if (args.size() > 0) { mean = *(args.begin());
                        if (args.size() > 1) { stddev = *(args.end() - 1); }
                    }
                    random_device dev;
                    default_random_engine gen(dev());
                    normal_distribution<T> dis(mean, stddev);
                    for (int i = 0; i < size; i++) { nums[i] = dis(gen); }
                }
                break;
            }
            case linalg::identity:
                for (int i = 0; i < size; i++) { nums[i] = 0; }
                for (int i = 0; i < size; i += rows + 1) { nums[i] = 1; }
                break;
            default:
                break;
            }
        }

        int get(int m, int n) { return nums[m * cols + n]; }
        void set(int m, int n, T val) { nums[m * cols + n] = val; }
        void print(string str = "", string str2 = "", bool newLines = true)
        {
            if (newLines) cout << endl;
            cout << str << " [[";
            for (int i = 0; i < size; i++)
            {
                cout << nums[i];
                if (i == size - 1) break;
                else if ((i + 1) % cols == 0) cout << "],\n [";
                else cout << ", ";
            }
            cout << "]]";
            cout << str2;
            if (newLines) cout << endl;
        }
        //********************************************************************************************************
        //*transposes matrix
        //*IN: none
        //*OUT: matrix with transposed values
        //*PRE: non-empty vector
        //*POST: new object on stack, move
        //*TEST: B1W1
        //********************************************************************************************************
        Matrix transposed()
        {
            Matrix newMat(cols, rows);
            if (cols == 1 || rows == 1)
            {
                for (int i = 0; i < newMat.size; i++)
                {
                    newMat.nums[i] = nums[i];
                }
            }
            else
            {
                /*for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < cols; c++) {
                        newMat[c][r] = (*this)[r][c];
                    }
                }*/
                for (int i = 0; i < size; i++)
                {
                    newMat.nums[i] = nums[(i * cols + i / newMat.cols) % size];
                }
            }
            return newMat;
        }
        T det(Matrix* mat = this)
        {
            if (rows != cols) return 0;
            //do something recursive
            return T;
        }
        Vector<T>* eigen()
        {
            Vector<T>* eigenVecs = new Vector<T>[cols];
            //do something, yes
            return eigenVecs;
        }
        T sum()
        {
            T res = 0;
            for (int i = 0; i < size; i++) {
                res += nums[i];
            }
            return res;
        }
        Vector<T> asVector()
        {
            Vector<T> newVec(size, nums);
            return newVec;
        }
       
        T* operator [](int m) { return &(nums[m * cols]); }
        Matrix operator + (Matrix mat)
        {
            Matrix newMat(rows, cols);
            for (int i = 0; i < size; i++) { newMat.nums[i] = nums[i] + (*mat.nums)[i]; }
            return newMat;
        }
        Matrix operator + (T a)
        {
            Matrix newMat(rows, cols);
            for (int i = 0; i < size; i++) { newMat.nums[i] = nums[i] + a; }
            return newMat;
        }
        Matrix operator - (Matrix mat)
        {
            Matrix newMat(rows, cols);
            for (int i = 0; i < size; i++) { newMat.nums[i] = nums[i] - (*mat.nums)[i]; }
            return newMat;
        }
        Matrix operator - (T a)
        {
            Matrix newMat(rows, cols);
            for (int i = 0; i < size; i++) { newMat.nums[i] = nums[i] - a; }
            return newMat;
        }
        Matrix operator * (Matrix mat)
        {
            Matrix newMat(rows, mat.cols);
            int x1, x2;
            for (int i = 0; i < newMat.size; i++)
            {
                newMat.nums[i] = 0;
                x1 = i / newMat.cols * cols;
                x2 = i % newMat.cols;
                for (int j = 0; j < cols; j++)
                {
                    newMat.nums[i] += nums[x1 + j] * mat.nums[x2 + j * mat.cols];
                }
            }
            return newMat;
        }
        Vector<T> operator * (Vector<T> vec)
        {
            int x = 0;
            Vector<T> newVec(rows);
            for (int i = 0; i < newVec.size; i++)
            {
                newVec.nums[i] = 0;
                x = i * cols;
                for (int j = 0; j < vec.size; j++)
                {
                    newVec.nums[i] += nums[x + j] * vec.nums[j];
                }
            }
            return newVec;
        }
        Matrix operator * (T a)
        {
            Matrix newMat(rows, cols);
            for (int i = 0; i < size; i++) { newMat.nums[i] = nums[i] * a; }
            return newMat;
        }
        Matrix operator / (T a)
        {
            Matrix newMat(rows, cols);
            for (int i = 0; i < size; i++) { newMat.nums[i] = nums[i] / a; }
            return newMat;
        }
    };
    template <class T>
    static Matrix<T>& operator += (Matrix<T>& X, Matrix<T>& Y) {
        for (int i = 0; i < X.size; i++) { (X.nums)[i] += (Y.nums)[i]; }
        return X;
    }
    template <class T>
    static Matrix<T>& operator -= (Matrix<T>& X, Matrix<T>& Y) {
        for (int i = 0; i < X.size; i++) { (X.nums)[i] -= (Y.nums)[i]; }
        return X;
    }
    template <class T>
    bool operator == (Matrix<T>& mat, Matrix<T>& mat2)
    {
        for (int i = 0; i < mat.size; i++) { if (mat.nums[i] != mat2.nums[i]) return false; }
        return true;
    }
    //template <class T>
    /*Matrix<T> operator * (Matrix<T>& mat1, Matrix<T>& mat2)
    {
        Matrix<T> newMat(mat1.rows, mat2.cols);
        int x1, x2;
        for (int i = 0; i < newMat.size; i++)
        {
            newMat.nums[i] = 0;
            x1 = i / newMat.cols * mat1.cols;
            x2 = i % newMat.cols;
            for (int j = 0; j < mat1.cols; j++)
            {
                newMat.nums[i] += mat1.nums[x1 + j] * mat2.nums[x2 + j * mat2.cols];
            }
        }
        return newMat;
    }*/
}
