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

void readInput(string path, vector <pair <Matrix, int> > &trainingData)
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
        trainingData.push_back(x);
    }
}

Matrix findMean(vector <pair <Matrix, int> > &trainingData, int classification)
{
    Matrix m(DIMEN, 1);
    int n = 0;
    for (int i = 0; i < trainingData.size(); ++i)
        if (trainingData[i].second == classification)
            m = m.add(trainingData[i].first, !(++n));
    for (int i = 0; i < DIMEN; ++i)
        m.A[i][0] /= n;
    return m;
}

Matrix intraCov(vector <pair <Matrix, int> > &trainingData, Matrix mean[])
{
    vector <Matrix> Sw;
    Matrix S(DIMEN, DIMEN);
    Sw.push_back(S);
    Sw.push_back(S);
    for (int i = 0; i < trainingData.size(); ++i)
    {
        Matrix m = trainingData[i].first.add(mean[trainingData[i].second], true);
        Sw[trainingData[i].second] = Sw[trainingData[i].second].add(m.multiply(m.transpose()));
    }
    return Sw[0].add(Sw[1]);
}

double calcEntropy(vector <pair<double, int> > r, int loc)
{
    int c[2] = {0, 0};
    double p;
    for (int i = 0; i < loc; ++i)
        c[r[i].second] += 1;
    p = c[0] / (double)(c[0] + c[1]);
    double entropy = loc * (0 - p * log2(p) - (1 - p) * log2(1 - p));
    c[0] = c[1] = 0;
    for (int i = loc; i < r.size(); ++i)
        c[r[i].second] += 1;
    p = c[0] / (double)(c[0] + c[1]);
    entropy += (r.size() - loc) * (0 - p * log2(p) - (1 - p) * log2(1 - p));
    return entropy;
}

int countPoints(vector <pair<double, int> > &r, int loc, bool higher, int classification)
{
    int counter = 0, start = 0, finish = r.size() - 1;
    if (higher) start = loc;
    else finish = loc - 1;
    for (int i = start; i <= finish; ++i)
        if (r[i].second == classification)
            ++counter;
    return counter;
}

int main()
{
    vector <pair <Matrix, int> >trainingData;
    readInput("../train.txt", trainingData);
    //Step 1: Find the center of each cluster
    Matrix mean[2];
    mean[0] = findMean(trainingData, 0);
    mean[1] = findMean(trainingData, 1);
    //Step 2: Find the covariance with each cluster
    Matrix Sw(DIMEN, DIMEN);
    Sw = intraCov(trainingData, mean);
    //Step 3: Find w, the weight vector
    Matrix w = Sw.inverse().multiply(mean[1].add(mean[0], true));
    //Step 4: project all the points to single dimension and sort them
    vector <pair<double, int> > wTx;
    for (int i = 0; i < trainingData.size(); ++i)
        wTx.push_back(pair <double, int> (w.transpose().multiply(trainingData[i].first).A[0][0], trainingData[i].second));
    sort(wTx.begin(), wTx.end());
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
    //Printing
    cout << countPoints(wTx, loc, false, 0) << " " << countPoints(wTx, loc, false, 1) << endl;
    cout << countPoints(wTx, loc, true, 0) << " " << countPoints(wTx, loc, true, 1) << endl;
}
