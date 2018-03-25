#include <bits/stdc++.h>
#define GAMMA 0.9
#define ETA 0.01

using namespace std;
typedef vector <pair <vector <double>, vector <double> > > Data;

class NN
{
    public:
    int layers;
    vector <int> sizes;
    vector <vector <vector <double> > > weights, delw, v;
    vector <vector <double> > z, delta;

    NN(vector <int> &sizes)
    {
        layers = sizes.size();
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
            v.push_back(vector <vector <double> > (sizes[i], vector <double> (sizes[i + 1])));
        }
    }

    int test(Data &testData)
    {
        int error = 0;
        for (int i = 0; i < testData.size(); ++i)
        {
            for (int j = 0; j < sizes[0]; ++j)
                z[0][j] = testData[i].first[j];
            forwardPass();
            if (distance(testData[i].second.begin(), max_element(testData[i].second.begin(), testData[i].second.end())) !=
                distance(z[layers - 1].begin(), max_element(z[layers - 1].begin(), z[layers - 1].end())))
                error += 1;
        }
        return error;
    }

    void train(Data &trainingData, Data &validationData, int batchSize, int maxIt)
    {
        int prev = validationData.size();
        for (int ep = 0; ep < maxIt; ++ep)
        {
            for (int batch = 0; batch < trainingData.size(); batch += batchSize)
            {
                for (int i = 0; i < layers - 1; ++i)
                    for (int j = 0; j < sizes[i]; ++j)
                        for (int k = 0; k < sizes[i + 1]; ++k)
                            delw[i][j][k] = 0;
                for (int i = 0; i < batchSize; ++i)
                {
                    for (int j = 0; j < sizes[0]; ++j)
                        z[0][j] = trainingData[batch + i].first[j];
                    backProp(trainingData[i].second);
                }
                for (int i = 0; i < layers - 1; ++i)
                    for (int j = 0; j < sizes[i]; ++j)
                        for (int k = 0; k < sizes[i + 1]; ++k)
                            delw[i][j][k] /= batchSize;
                gradDescent();
            }
            if (test(validationData) > prev)
                break;
            else
                prev = test(validationData);
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

    double sigmoid(double x)
    {
        return 1 / (1 + exp(-x));
    }

    void backProp(vector <double> &t)
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

void readInput(string path, Data &data, int dimenX, int dimenY)
{
    ifstream f(path, ios::in);
    string line;
    while(getline(f, line))
    {
        vector <double> x(dimenX), y(dimenY);
        int y_pos;
        stringstream ss(line);
        for (int j = 0; j < dimenX; ++j, ss.ignore())
            ss >> x[j];
        ss >> y_pos;
        y[y_pos] = 1;
        data.push_back(pair <vector <double> , vector <double> > (x, y));
    }
}

int main()
{
    Data trainingData, validationData, testData;
    readInput("../train.txt", trainingData, 64, 10);
    readInput("../test.txt", testData, 64, 10);
    readInput("../validation.txt", validationData, 64, 10);
    vector <int> sizes;
    sizes.push_back(64); sizes.push_back(5); sizes.push_back(10);
    NN network(sizes);
    network.train(trainingData, validationData, 100, 3000);
    cout << network.test(testData);
    return 0;
}
