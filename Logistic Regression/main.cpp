#include <bits/stdc++.h>
#define DIMEN 4

using namespace std;

class Matrix
{
    public:
    vector <vector <double> > A;
    Matrix(int r, int c)
    {
        vector <double> temp(c);
        while (r--)
            A.push_back(temp);
    }
    Matrix(){}
    Matrix add(Matrix m, bool invert = false)
    {
        Matrix dif(A.size(), A[0].size());
        if (A.size() != m.A.size() || A[0].size() != m.A[0].size()) return dif;
        for (int i = 0; i < A.size(); ++i)
            for (int j = 0; j < A[0].size(); ++j)
                dif.A[i][j] = A[i][j] + (invert? -1: 1) * m.A[i][j];
        return dif;
    }
    Matrix multiply(Matrix m)
    {
        Matrix prod(A.size(), m.A[0].size());
        if (A[0].size() != m.A.size()) return prod;
        for (int i = 0; i < A.size(); ++i)
            for (int j = 0; j < m.A[0].size(); ++j)
                for (int k = 0; k < m.A.size(); ++k)
                    prod.A[i][j] += A[i][k] * m.A[k][j];
        return prod;
    }
    Matrix multiply(double x)
    {
        Matrix B(A.size(), A[0].size());
        for (int i = 0; i < A.size(); ++i)
            for (int j = 0; j < A[0].size(); ++j)
                B[i][j] = A[i][j] * x;
        return B;
    }
    double sum()
    {
        double sum = 0;
        for (int i = 0; i < A.size(); ++i)
            for (int j = 0; j < A[0].size(); ++j)
                sum += A[i][j];
        return sum;
    }
    double det()
    {
        if (A.size() != A[0].size()) return 0;
        if (A.size() == 1) return A[0][0];
        double determinant = 0;
        for (int i = 0; i < A.size(); ++i)
        {
            Matrix m(A.size() - 1, A.size() - 1);
            for (int j = 0; j < A.size() - 1; ++j)
                for (int k = 0; k < A.size() - 1; ++k)
                    m.A[j][k] = A[j + 1][k >= i? k + 1: k];
            determinant += (i % 2? -1: 1) * A[0][i] * m.det();
        }
        return determinant;
    }
    double cofactor(int r, int c)
    {
        if (A.size() != A[0].size()) return 0;
        Matrix m(A.size() - 1, A.size() - 1);
        for (int j = 0; j < A.size() - 1; ++j)
            for (int k = 0; k < A.size() - 1; ++k)
                m.A[j][k] = A[j >= r? j + 1: j][k >= c? k + 1: k];
        return m.det();
    }
    Matrix inverse()
    {
        Matrix adj(A.size(), A.size());
        double determinant = det();
        if (!determinant) return adj;
        for (int i = 0; i < A.size(); ++i)
            for (int j = 0; j < A.size(); ++j)
                adj.A[j][i] = ((i + j % 2)? -1: 1) * cofactor(i, j) / determinant;
        return adj;
    }
    Matrix transpose()
    {
        Matrix t(A[0].size(), A.size());
        for (int i = 0; i < A.size(); ++i)
            for (int j = 0; j < A[0].size(); ++j)
                t.A[j][i] = A[i][j];
        return t;
    }
};

void readInput(string path, vector <pair <Matrix, int> > &Data)
{
    ifstream f(path, ios::in);
    string line;
    pair <Matrix, int> x;
    x.first = *new Matrix(DIMEN, 1);
    while(getline(f, line))
    {
        stringstream ss(line);
        for (int j = 0; j < DIMEN; ++j, ss.ignore())
            ss >> x.first.A[j][0];
        ss >> x.second;
        Data.push_back(x);
    }
}

int countPoints(vector <pair<double, int> > &r, double y0, bool higher, int classification)
{
    int counter = 0;
    for (int i = 0; i < r.size(); ++i)
        if (r[i].second == classification && (r[i].first >= y0 == higher))
            ++counter;
    return counter;
}

int main()
{
    vector <pair <Matrix, int> >trainingData, testingData;
    readInput("../train.txt", trainingData);
    //Step 1: Initialize w
    Matrix w(DIMEN, 1);
    //Step 2: Gradient descent
    double eta = 0.01, error = 1;
    while (error > 0.001)
    {
        Matrix delE(DIMEN, 1);
        for (int i = 0; i < trainingData.size(); ++i)
        {
            double yn = sigmoid(w.transpose().multiply(trainingData[i].first).A[0][0]);
            delE = delE.add(trainingData[i].first.multiply(yn - trainingData[i].second));
        }
        error = delE.sum();
        w = w.add(delE.multiply(eta), true);
    }


    //Step 4: project all the points to single dimension
    vector <pair<double, int> > wTx;
    for (int i = 0; i < trainingData.size(); ++i)
        wTx.push_back(pair <double, int> (sigmoid(w.transpose().multiply(trainingData[i].first).A[0][0]), trainingData[i].second));
    //Step 5: calc y0
    double minEntropy = 10e7;
    int loc = 0;
    for (int i = 1; i < trainingData.size(); ++i)
    {
        double E = calcEntropy(wTx, i);
        minEntropy = min(minEntropy, E);
        if (minEntropy == E)
            loc = i;
    }
    double y0 = (wTx[loc].first + wTx[loc + 1].first) / 2;
    //Testing training data
    cout << countPoints(wTx, y0, false, 0) << " " << countPoints(wTx, y0, false, 1) << endl;
    cout << countPoints(wTx, y0, true, 0) << " " << countPoints(wTx, y0, true, 1) << endl;
    //Testing test data
    readInput("../test.txt", testingData);
    vector <pair<double, int> > r;
    for (int i = 0; i < testingData.size(); ++i)
        r.push_back(pair <double, int> (w.transpose().multiply(testingData[i].first).A[0][0], testingData[i].second));
    cout << countPoints(r, y0, false, 0) << " " << countPoints(r, y0, false, 1) << endl;
    cout << countPoints(r, y0, true, 0) << " " << countPoints(r, y0, true, 1) << endl;

}
