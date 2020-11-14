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

#define NUMERIC_ONLY(T) T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type

namespace linalg
{
    //The way a matrix or std::vector should be initialized. The constructor takes an initType, along with a list of optional arguments.
    //zeros: set each element to 0
    //ones: set each element to 1
    //number: set each element to (first arg)
    //identity: set each element to 1 on the diagonal and 0 everywhere else
    //uniform: matrix with each element on a uniform distribution from (first arg) to (second arg)
    //normal: matrix with each element on a normal distribution with mean (first arg) and standard deviation (second arg)
    enum initType { zeros, ones, number, identity, uniform, normal };
    template <class NUMERIC_ONLY(T)>
    class Matrix;
    template <class NUMERIC_ONLY(T)>
    class Vector
    {
    public:
        std::vector<T> nums;
        size_t size;

        Vector()
        {
            size = 0;
            nums = std::vector<T>();
        }
        Vector(int s, T* varr = NULL)
        {
            size = s;
            if (varr != NULL) { nums.assign(varr, varr + size); }
            else nums = std::vector<T>(size);
        }
        Vector(std::vector<T> varr)
        {
            size = varr.size();
            nums = varr;
        }
        Vector(Matrix<T>& mat)
        {
            size = mat.size;
            nums = mat.nums;
        }
        Vector(std::initializer_list<T> initNums)
        {
            size = distance(initNums.begin(), initNums.end());
            nums = std::vector<T>(size);
            int i = 0;
            for (T num : initNums) { nums[i] = num; i++; }
        }
        Vector(int s, std::function<T(int)> initFunc)
        {
            size = s;
            nums = std::vector<T>(size);
            for (int i = 0; i < size; i++) { nums[i] = initFunc(i); }
        }
        Vector(int s, initType type, std::initializer_list<T> args = {})
        {
            size = s;
            nums = std::vector<T>(size);
        retryInitType:
            switch (type)
            {
            case linalg::zeros:
                for (int i = 0; i < size; i++) { nums[i] = 0; }
                break;
            case linalg::ones:
                for (int i = 0; i < size; i++) { nums[i] = 1; }
                break;
            case linalg::number:
                if (args.size() > 0)
                {
                    T num = *(args.begin());
                    for (int i = 0; i < size; i++) { nums[i] = num; }
                }
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
                    std::random_device dev;
                    std::default_random_engine gen(dev());
                    std::uniform_real_distribution<T> dis(min, max);
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
                    std::random_device dev;
                    std::default_random_engine gen(dev());
                    std::normal_distribution<T> dis(mean, stddev);
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

        void print(std::string str = "", std::string str2 = "", bool newLines = true)
        {
            if (newLines) std::cout << std::endl;
            std::cout << str << " [" << nums[0];
            for (int i = 1; i < size; i++) { std::cout << ", " << nums[i]; }
            std::cout << "]";
            std::cout << str2;
            if (newLines) std::cout << std::endl;
        }
        void print(int elems)
        {
            if (elems > size) elems = size;
            std::cout << "\n[" << nums[0];
            for (int i = 1; i < elems / 2; i++) { std::cout << ", " << nums[i]; }
            std::cout << ", . . . ";
            for (int i = size - elems / 2; i < size; i++) { std::cout << ", " << nums[i]; }
            std::cout << "]\n";
        }
        T dot(Vector vec)
        {
            T res = 0;
            for (int i = 0; i < size; i++) { res += nums[i] * vec.nums[i]; }
            return res;
        }
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
            Matrix<T> newMat(m, n, *this);
            return newMat;
        }

        T& operator [] (int i) { return nums[i]; }
    };
    template <class NUMERIC_ONLY(T)>
    class Matrix
    {
    public:
        std::vector<T> nums;
        int rows;
        int cols;
        int size;
        
        Matrix()
        {
            rows = 0;
            cols = 0;
            size = 0;
            nums = std::vector<T>();
        }
        Matrix(int m, int n, T* marr = NULL)
        {
            rows = m;
            cols = n;
            size = m * n;
            if (marr != NULL) { nums.assign(marr, marr + size); }
            else nums = std::vector<T>(size);
        }
        Matrix(int m, int n, std::vector<T> marr)
        {
            rows = m;
            cols = n;
            size = m * n;
            nums = marr;
        }
        Matrix(int m, int n, Vector<T>& vec)
        {
            rows = m;
            cols = n;
            size = m * n;
            nums = vec.nums;
        }
        Matrix(int m, int n, std::initializer_list<T> initNums)
        {
            rows = m;
            cols = n;
            size = m * n;
            nums = std::vector<T>(size);
            int i = 0;
            for (T num : initNums) { nums[i] = num; i++; }
        }
        Matrix(int m, int n, std::function<T(int)> initFunc)
        {
            rows = m;
            cols = n;
            size = m * n;
            nums = std::vector<T>(size);
            for (int i = 0; i < size; i++) { nums[i] = initFunc(i); }
        }
        Matrix(int m, int n, initType type, std::initializer_list<T> args = {})
        {
            rows = m;
            cols = n;
            size = m * n;
            nums = std::vector<T>(size);
            switch (type)
            {
            case linalg::zeros:
                for (int i = 0; i < size; i++) { nums[i] = 0; }
                break;
            case linalg::ones:
                for (int i = 0; i < size; i++) { nums[i] = 1; }
                break;
            case linalg::number:
                if (args.size() > 0)
                {
                    T num = *(args.begin());
                    for (int i = 0; i < size; i++) { nums[i] = num; }
                }
                break;
            case linalg::uniform:
            {
                if (typeid(T) == typeid(float) || typeid(T) == typeid(double) || typeid(T) == typeid(float))
                {
                    T min = -1, max = 1;
                    if (args.size() > 0) { min = *(args.begin());
                        if (args.size() > 1) { max = *(args.end() - 1); } }
                    std::random_device dev;
                    std::default_random_engine gen(dev());
                    std::uniform_real_distribution<T> dis(min, max);
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
                    std::random_device dev;
                    std::default_random_engine gen(dev());
                    std::normal_distribution<T> dis(mean, stddev);
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

        void print(std::string str = "", std::string str2 = "", bool newLines = true)
        {
            if (newLines) std::cout << std::endl;
            std::cout << str << " [[";
            for (int i = 0; i < size; i++)
            {
                std::cout << nums[i];
                if (i == size - 1) break;
                else if ((i + 1) % cols == 0) std::cout << "],\n [";
                else std::cout << ", ";
            }
            std::cout << "]]";
            std::cout << str2;
            if (newLines) std::cout << std::endl;
        }
        void print(int elems)
        {
            if (elems > size) elems = size;
            std::cout << "\n[[";
            for (int i = 0; i < elems / 2; i++)
            {
                std::cout << nums[i];
                if ((i + 1) % cols == 0) std::cout << "],\n [";
                else std::cout << ", ";
            }
            std::cout << " . . . ],\n . . . \n[. . . , ";
            for (int i = size - elems / 2; i < size; i++)
            {
                std::cout << nums[i];
                if (i == size - 1) break;
                else if ((i + 1) % cols == 0) std::cout << "],\n [";
                else std::cout << ", ";
            }
            std::cout << "]]\n";
        }
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
            else {
                for (int i = 0; i < newMat.rows; i++) {
                    for (int j = 0; j < newMat.cols; j++) {
                        newMat[i][j] = (*this)[j][i];
                    }
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
            Vector<T> newVec(*this);
            return newVec;
        }
       
        T* operator [](int m) { return &(nums[m * cols]); }
    };

    /// Matrix operations ///
    template <class T>
    bool operator == (const Matrix<T>& X, const Matrix<T>& Y)
    {
        for (int i = 0; i < X.size; i++) { if (X.nums[i] != Y.nums[i]) return false; }
        return true;
    }

    template <class T>
    Matrix<T> operator + (const Matrix<T>& X, const Matrix<T>& Y)
    {
        Matrix<T> newMat(X.rows, X.cols);
        for (int i = 0; i < X.size; i++) { newMat.nums[i] = X.nums[i] + Y.nums[i]; }
        return newMat;
    }
    template <class T, class NUMERIC_ONLY(U)>
    Matrix<T> operator + (const Matrix<T>& X, U a)
    {
        Matrix<T> newMat(X.rows, X.cols);
        for (int i = 0; i < X.size; i++) { newMat.nums[i] = X.nums[i] + a; }
        return newMat;
    }
    template <class T>
    Matrix<T>& operator += (Matrix<T>& X, const Matrix<T>& Y) {
        for (int i = 0; i < X.size; i++) { (X.nums)[i] += (Y.nums)[i]; }
        return X;
    }
    template <class T, class NUMERIC_ONLY(U)>
    Matrix<T>& operator += (Matrix<T>& X, U a)
    {
        for (int i = 0; i < X.size; i++) { X.nums[i] += a; }
        return X;
    }

    template <class T>
    Matrix<T> operator - (const Matrix<T>& X, const Matrix<T>& Y)
    {
        Matrix<T> newMat(X.rows, X.cols);
        for (int i = 0; i < X.size; i++) { newMat.nums[i] = X.nums[i] - Y.nums[i]; }
        return newMat;
    }
    template <class T, class NUMERIC_ONLY(U)>
    Matrix<T> operator - (const Matrix<T>& X, U a)
    {
        Matrix<T> newMat(X.rows, X.cols);
        for (int i = 0; i < X.size; i++) { newMat.nums[i] = X.nums[i] - a; }
        return newMat;
    }
    template <class T>
    Matrix<T>& operator -= (Matrix<T>& X, const Matrix<T>& Y) {
        for (int i = 0; i < X.size; i++) { (X.nums)[i] -= (Y.nums)[i]; }
        return X;
    }
    template <class T, class NUMERIC_ONLY(U)>
    Matrix<T>& operator -= (Matrix<T>& X, U a)
    {
        for (int i = 0; i < X.size; i++) { X.nums[i] -= a; }
        return X;
    }

    template <class T>
    Matrix<T> operator * (const Matrix<T>& X, const Matrix<T>& Y)
    {
        Matrix<T> newMat(X.rows, Y.cols);
        int x1, x2;
        for (int i = 0; i < newMat.size; i++)
        {
            newMat.nums[i] = 0;
            x1 = i / newMat.cols * X.cols;
            x2 = i % newMat.cols;
            for (int j = 0; j < X.cols; j++)
            {
                newMat.nums[i] += X.nums[x1 + j] * Y.nums[x2 + j * Y.cols];
            }
        }
        return newMat;
    }
    template <class T, class NUMERIC_ONLY(U)>
    Matrix<T> operator * (const Matrix<T>& X, U a)
    {
        Matrix<T> newMat(X.rows, X.cols);
        for (int i = 0; i < X.size; i++) { newMat.nums[i] = X.nums[i] * a; }
        return newMat;
    }
    template <class T>
    Vector<T> operator * (const Matrix<T>& X, const Vector<T>& vec)
    {
        int x = 0;
        Vector<T> newVec(X.rows);
        for (int i = 0; i < newVec.size; i++)
        {
            newVec.nums[i] = 0;
            x = i * X.cols;
            for (int j = 0; j < vec.size; j++)
            {
                newVec.nums[i] += X.nums[x + j] * vec.nums[j];
            }
        }
        return newVec;
    }
    template <class T, class NUMERIC_ONLY(U)>
    Matrix<T>& operator *= (Matrix<T>& X, U a) {
        for (int i = 0; i < X.size; i++) { X.nums[i] *= a; }
        return X;
    }

    /// Vector operations ///
    template <class T>
    bool operator == (const Vector<T>& X, const Vector<T>& Y)
    {
        for (int i = 0; i < X.size; i++) { if (X.nums[i] != Y.nums[i]) return false; }
        return true;
    }

    template <class T>
    Vector<T> operator + (const Vector<T>& X, const Vector<T>& Y)
    {
        Vector<T> newVec(X.size);
        for (int i = 0; i < X.size; i++) { newVec.nums[i] = X.nums[i] + Y.nums[i]; }
        return newVec;
    }
    template <class T, class NUMERIC_ONLY(U)>
    Vector<T> operator + (const Vector<T>& X, U a)
    {
        Vector<T> newVec(X.size);
        for (int i = 0; i < X.size; i++) { newVec.nums[i] = X.nums[i] + a; }
        return newVec;
    }
    template <class T>
    Vector<T>& operator += (Vector<T>& X, const Vector<T>& Y) {
        for (int i = 0; i < X.size; i++) { (X.nums)[i] += (Y.nums)[i]; }
        return X;
    }
    template <class T, class NUMERIC_ONLY(U)>
    Vector<T>& operator += (Vector<T>& X, U a)
    {
        for (int i = 0; i < X.size; i++) { X.nums[i] += a; }
        return X;
    }

    template <class T>
    Vector<T> operator - (const Vector<T>& X, const Vector<T>& Y)
    {
        Vector<T> newVec(X.size);
        for (int i = 0; i < X.size; i++) { newVec.nums[i] = X.nums[i] - Y.nums[i]; }
        return newVec;
    }
    template <class T, class NUMERIC_ONLY(U)>
    Vector<T> operator - (const Vector<T>& X, U a)
    {
        Vector<T> newVec(X.size);
        for (int i = 0; i < X.size; i++) { newVec.nums[i] = X.nums[i] - a; }
        return newVec;
    }
    template <class T>
    Vector<T>& operator -= (Vector<T>& X, const Vector<T>& Y) {
        for (int i = 0; i < X.size; i++) { (X.nums)[i] -= (Y.nums)[i]; }
        return X;
    }
    template <class T, class NUMERIC_ONLY(U)>
    Vector<T>& operator -= (Vector<T>& X, U a)
    {
        for (int i = 0; i < X.size; i++) { X.nums[i] -= a; }
        return X;
    }

    template <class T>
    T operator * (const Vector<T>& X, const Vector<T>& Y)
    {
        T res = 0;
        for (int i = 0; i < X.size; i++) { res += X.nums[i] * Y.nums[i]; }
        return res;
    }
    template <class T, class NUMERIC_ONLY(U)>
    Vector<T> operator * (const Vector<T>& X, U a)
    {
        Vector<T> newVec(X.size);
        for (int i = 0; i < X.size; i++) { newVec.nums[i] = X.nums[i] * a; }
        return newVec;
    }
    template <class T, class NUMERIC_ONLY(U)>
    Vector<T>& operator *= (Vector<T>& X, U a) {
        for (int i = 0; i < X.size; i++) { X.nums[i] *= a; }
        return X;
    }
}
