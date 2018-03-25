#include <bits/stdc++.h>
#define GAMMA 0.9
#define ETA 0.01

using namespace std;
typedef vector <pair <vector <double>, int> > Data

class NN
{
    int layers;
    vector <int> sizes;
    vector <vector <vector <double> > > weights, delw, v;
    vector <vector <double> > z, delta;

    NN(int layers, vector <int> &sizes)
    {
        this->layers = layers;
        for (int i = 0; i < layers; ++i)
        {
            this->sizes[i] = sizes[i];
            delta.push_back(vector <double> (sizes[i]));
            z.push_back(vector <double> (sizes[i]));
        }
        for (int i = 0; i < layers - 1; ++i)
        {
            weights.push_back(vector <vector <double> > (sizes[i], vector <double> (sizes[i + 1])));
            delw.push_back(vector <vector <double> > (sizes[i], vector <double> (sizes[i + 1])));
            v.push_back(vector <vector <double> > (sizes[i], vector <double> (sizes[i + 1])))
        }
    }

    void train(Data &trainingData, Data &validationData, int batchSize, int maxIt)
    {
        for (int batch = 0; batch < trainingData.size(); batch += batchSize)
        {
            for (int i = 0; i < batchSize; ++i)
            {
                for (int j = 0; j < sizes[0]; ++j)
                    z[0][j] = trainingData[batch + i].first[j];
                backProp();
            }
            gradDescent();
        }
    }

    void forwardPass()
    {
        for (int i = 1; i < layers; ++i)
        {
            for (int j = 0; j < sizes[i]; ++j)
            {
                z[i][j] = 0;
                for (int k = 0; k < sizes[i - 1]; ++k)
                    z[i][j] += weights[i - 1][k][j] * z[i - 1][k];
                z[i][j] = sigmoid(z[i][j]);
            }
        }
    }

    void gradDescent()
    {
        for (int i = 0; i < layers - 1; ++i)
        {
            for (int j = 0; j < sizes[i]; ++j)
            {
                for (int k = 0; k < sizes[i + 1]; ++k)
                {
                    v[i][j][k] = GAMMA * v[i][j][k] + ETA * delw[i][j][k];
                    weights[i][j][k] -= v[i][j][k];
                }
            }
        }
    }

    void backProp()
    {
        forwardPass();
        for (int i = 0; i < sizes[layers - 1]; ++i)
            delta[layers - 1][i] = z[layers - 1][i] - t[i];
        for (int i = layers - 2; i >= 0; --i)
        {
            for (int j = 0; j < sizes[i]; ++j)
            {
                delta[i][j] = 0;
                for (int k = 0; k < sizes[i + 1]; ++k)
                    delta[i][j] += weights[i + 1][j][k] * delta[i + 1][k];
                delta[i][j] = sigmoid(z[i][j]) * (1 - sigmoid(z[i][j])) * delta[i][j];
            }
        }
        for (int i = 0; i < layers - 1; ++i)
            for (int j = 0; j < sizes[i]; ++j)
                for (int k = 0; k < sizes[i + 1]; ++k)
                    delw[i][j][k] += delta[i + 1][k] * z[i][j];
    }
};

int main()
{
    Matrix m1(2, 2), m2(2, 2);
    m1.A[0][0] = 0; m1.A[0][1] = 1;
    m1.A[1][0] = 2; m1.A[1][1] = 3;
    m2.A[0][0] = 1; m2.A[0][1] = 2;
    m2.A[1][0] = 1; m2.A[1][1] = 2;
    m1.binaryOp(m2, plus<double>());
    cout << m1.A[0][0] << " " << m1.A[0][1] << endl;
    cout << m1.A[1][0] << " " << m1.A[1][1] << endl;
    return 0;
}
