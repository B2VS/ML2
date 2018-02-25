#include <bits/stdc++.h>

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
    Matrix subtract(Matrix &m)
    {
        Matrix dif(A.size(), A[0].size());
        if (A.size() != m.A.size() || A[0].size() != m.A[0].size()) return dif;
        for (int i = 0; i < A.size(); ++i)
            for (int j = 0; j < A[0].size(); ++j)
                dif.A[i][j] = A[i][j] - m.A[i][j];
        return dif;
    }
    Matrix multiply(Matrix &m)
    {
        Matrix prod(A.size(), m.A[0].size());
        if (A[0].size() != m.A.size()) return prod;
        for (int i = 0; i < A.size(); ++i)
            for (int j = 0; j < m.A[0].size(); ++j)
                for (int k = 0; k < m.A.size(); ++k)
                    prod.A[i][j] += A[i][k] * m.A[k][j];
        return prod;
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
};

int main()
{
    Matrix m(4, 4);
    m.A[0][0] = 3; m.A[0][1] = 2; m.A[0][2] = 8; m.A[0][3] = 4;
    m.A[1][0] = 7; m.A[1][1] = 12; m.A[1][2] = 4; m.A[1][3] = 6;
    m.A[2][0] = 1; m.A[2][1] = 1; m.A[2][2] = 1; m.A[2][3] = 3;
    m.A[3][0] = 4; m.A[3][1] = 3; m.A[3][2] = 2; m.A[3][3] = 1;
    cout << m.det();

    /*
    vector <pair <Matrix, int> >trainingData;
    readInput("../test.txt", trainingData);
    Matrix m0, m1;
    m0 = findMean(trainingData, 0);
    m1 = findMean(trainingData, 1);
    int dimen = trainingData[0].first.size();
    double Sw[dimen][dimen];
    intraCov(trainingData, Sw);
    Matrix w = Sw.inverse().multiply(m1.subtract(m0));*/
}
