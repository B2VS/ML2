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
                B.A[i][j] = A[i][j] * x;
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
    x.first = *new Matrix(DIMEN + 1, 1);
    x.first.A[0][0] = 1;
    while(getline(f, line))
    {
        stringstream ss(line);
        for (int j = 1; j <= DIMEN; ++j, ss.ignore())
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

double sigmoid(double x)
{
    return 1 / (1 + exp(0 - x));
}

int main()
{
    vector <pair <Matrix, int> >trainingData, testingData;
    readInput("../train.txt", trainingData);
    //Step 1: Initialize w
    cout << "Initializing w with zeros..." << endl;
    Matrix w(DIMEN + 1, 1);
    //Step 2: Gradient descent
    cout << "Finding w with error less than 0.01 using gradient descent..." << endl;
    double eta = 0.001, error = 1;
    while (error > 0.01)
    {
        Matrix delE(DIMEN + 1, 1);
        for (int i = 0; i < trainingData.size(); ++i)
        {
            double yn = sigmoid(w.transpose().multiply(trainingData[i].first).A[0][0]);
            delE = delE.add(trainingData[i].first.multiply(yn - trainingData[i].second));
        }
        error = abs(delE.sum());
        w = w.add(delE.multiply(eta), true);
    }
    //Step 3: project all the points to single dimension
    cout << "Projecting all points to 1D..." << endl;
    vector <pair<double, int> > y;
    for (int i = 0; i < trainingData.size(); ++i)
        y.push_back(pair <double, int> (w.transpose().multiply(trainingData[i].first).A[0][0], trainingData[i].second));
    //Testing training data
    cout << "Training Data: " << endl;
    cout << "   " << countPoints(y, 0, true, 1) << " " << countPoints(y, 0, true, 0) << endl;
    cout << "   " << countPoints(y, 0, false, 1) << " " << countPoints(y, 0, false, 0) << endl;
    //Testing test data
    readInput("../test.txt", testingData);
    vector <pair<double, int> > r;
    for (int i = 0; i < testingData.size(); ++i)
        r.push_back(pair <double, int> (w.transpose().multiply(testingData[i].first).A[0][0], testingData[i].second));
    cout << "Testing Data: " << endl;
    double tp = countPoints(r, 0, true, 1);
    double tn = countPoints(r, 0, false, 0);
    double fp = countPoints(r, 0, true, 0);
    double fn = countPoints(r, 0, false, 1);
    cout << "   " << tp << " " << fp << endl;
    cout << "   " << fn << " " << tn << endl;
    cout << "Accuracy: " << (tp + tn) / (tp + tn + fp + fn) << endl;
    cout << "Precision: " << tp / (tp + fp) << endl;
    cout << "Recall: " << tp / (tp + fn) << endl;
}
