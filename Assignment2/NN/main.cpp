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
    vector <vector <vector <double> > > weights, delw, vw;
    vector <vector <double> > z, delta, bias, delb, vb;

    NN(vector <int> &sizes)
    {
        layers = sizes.size();
        for (int i = 0; i < layers; ++i)
        {
            this->sizes.push_back(sizes[i]);
            delta.push_back(vector <double> (sizes[i]));
            z.push_back(vector <double> (sizes[i]));
            bias.push_back(vector <double> (sizes[i]));
            vb.push_back(vector <double> (sizes[i]));
            delb.push_back(vector <double> (sizes[i]));
        }
        for (int i = 0; i < layers - 1; ++i)
        {
            weights.push_back(vector <vector <double> > (sizes[i], vector <double> (sizes[i + 1])));
            delw.push_back(vector <vector <double> > (sizes[i], vector <double> (sizes[i + 1])));
            vw.push_back(vector <vector <double> > (sizes[i], vector <double> (sizes[i + 1])));
        }
        randomWeights();
    }

    void randomWeights()
    {
        for (int i = 0; i < layers - 1; ++i)
            for (int j = 0; j < sizes[i]; ++j)
                for (int k = 0; k < sizes[i + 1]; ++k)
                    weights[i][j][k] = -1 + static_cast <float> (rand()) / static_cast <float> (RAND_MAX / 2);
        for (int i = 0; i < layers; ++i)
            for (int j = 0; j < sizes[i]; ++j)
                bias[i][j] = -1 + static_cast <float> (rand()) / static_cast <float> (RAND_MAX / 2);
    }

    int test(Data &testData)
    {
        int error = 0;
        for (int i = 0; i < testData.size(); ++i)
        {
            for (int j = 0; j < sizes[0]; ++j)
                z[0][j] = testData[i].first[j];
            forwardPass();
            //cout << "(" << distance(testData[i].second.begin(), max_element(testData[i].second.begin(), testData[i].second.end()));
            //cout << "," << distance(z[layers - 1].begin(), max_element(z[layers - 1].begin(), z[layers - 1].end())) << ") ";
            if (distance(testData[i].second.begin(), max_element(testData[i].second.begin(), testData[i].second.end())) !=
                distance(z[layers - 1].begin(), max_element(z[layers - 1].begin(), z[layers - 1].end())))
                error += 1;
        }
        cout << error << "/" << testData.size();
        return error;
    }

    void train2(Data &trainingData)
    {
        for (int ep = 0; ep < 1000; ++ep)
        {
            random_shuffle(trainingData.begin(), trainingData.end());
            for (int i = 0; i < trainingData.size(); ++i)
            {
                backProp(trainingData[i].second);
                gradDescent();
            }
            test(trainingData);
        }
    }

    void train(Data &trainingData, Data &validationData, int batchSize, int maxIt)
    {
        //int prev = validationData.size();
        for (int ep = 0; ep < maxIt; ++ep)
        {
            cout << endl << "ep = " << ep << ": ";
            random_shuffle(trainingData.begin(), trainingData.end());
            for (int batch = 0; batch < trainingData.size(); batch += batchSize)
            {
                for (int i = 0; i < layers - 1; ++i)
                    for (int j = 0; j < sizes[i]; ++j)
                        for (int k = 0; k < sizes[i + 1]; ++k)
                            delw[i][j][k] = 0;
                for (int i = 0; i < layers; ++i)
                    for (int j = 0; j < sizes[i]; ++j)
                        delb[i][j] = 0;
                //cout << "c0" << endl;
                for (int i = 0; i < batchSize; ++i)
                {
                    for (int j = 0; j < sizes[0]; ++j)
                        z[0][j] = trainingData[batch + i].first[j];
                    //cout << "c+" << i << endl;
                    backProp(trainingData[batch + i].second);
                    /*for (int j = 0; j < sizes[layers - 1]; ++j)
                        cout << z[layers - 1][j] << " ";
                    cout << endl;
                    for (int j = 0; j < sizes[layers - 1]; ++j)
                        cout << trainingData[batch + i].second[j] << " ";
                    cout << endl;*/
                }
                //cout << "c1" << endl;
                for (int i = 0; i < layers - 1; ++i)
                    for (int j = 0; j < sizes[i]; ++j)
                        for (int k = 0; k < sizes[i + 1]; ++k)
                            delw[i][j][k] /= batchSize;
                for (int i = 0; i < layers; ++i)
                    for (int j = 0; j < sizes[i]; ++j)
                        delb[i][j] /= batchSize;
                gradDescent();
            }
            int errors = test(validationData);
            /*if (errors > prev)
                break;
            else
                prev = errors;*/
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
                z[i][j] = sigmoid(z[i][j] + bias[i][j]);
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
                    vw[i][j][k] = GAMMA * vw[i][j][k] + ETA * delw[i][j][k];
                    weights[i][j][k] -= vw[i][j][k];
                }
            }
        }
        for (int i = 0; i < layers; ++i)
        {
            for (int j = 0; j < sizes[i]; ++j)
            {
                vb[i][j] = GAMMA * vb[i][j] + ETA * delta[i][j];
                bias[i][j] -= vb[i][j];
            }
        }
    }

    double sigmoid(double x)
    {
        return 1 / (1 + exp(-x));
    }

    void backProp(vector <double> &t)
    {
        //cout << "b0" << endl;
        forwardPass();
        //cout << "b1" << endl;
        for (int i = 0; i < sizes[layers - 1]; ++i)
            delta[layers - 1][i] = z[layers - 1][i] - t[i];
        //cout << "b2" << endl;
        for (int i = layers - 2; i > 0; --i)
        {
            for (int j = 0; j < sizes[i]; ++j)
            {
                delta[i][j] = 0;
                //cout << "b3" << endl;
                for (int k = 0; k < sizes[i + 1]; ++k)
                    delta[i][j] += weights[i][j][k] * delta[i + 1][k];
                //cout << "b4" << endl;
                delta[i][j] = z[i][j] * (1 - z[i][j]) * delta[i][j];
            }
            //cout << "b5" << endl;
        }
        for (int i = 0; i < layers - 1; ++i)
            for (int j = 0; j < sizes[i]; ++j)
                for (int k = 0; k < sizes[i + 1]; ++k)
                    delw[i][j][k] += delta[i + 1][k] * z[i][j];
        for (int i = 0; i < layers; ++i)
            for (int j = 0; j < sizes[i]; ++j)
                delb[i][j] += delta[i][j];
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
    srand(time(NULL));
    Data trainingData, validationData, testData;
    readInput("../train.txt", trainingData, 64, 10);
    readInput("../test.txt", testData, 64, 10);
    readInput("../validation.txt", validationData, 64, 10);
    cout << trainingData.size() << endl;
    cout << "input done" << endl;
    vector <int> sizes;
    sizes.push_back(64); sizes.push_back(5); sizes.push_back(10);
    NN network(sizes);
    cout << "nn made" << endl;
    network.train(trainingData, validationData, 100, 3000);
    //network.train2(trainingData);
    cout << network.test(testData);
    return 0;
}
