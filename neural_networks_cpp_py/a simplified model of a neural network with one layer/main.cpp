#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double deriv_sigmoid(double x) {
    return x * (1 - x);
}

std::vector<std::vector<double>> x_data = {
    {180, 120}, {180, 60}, {175, 110}, {170, 50},
};

std::vector<std::vector<double>> y_data = {
    {1, 0}, {0, 1}, {1, 0}, {0, 1},
};

class NeuralNetwork {
public:
    double learn_rate;
    std::vector<std::vector<double>> w1;
    double b1;

    NeuralNetwork(int input_dim, int output_dim) {
        learn_rate = 0.01; 
        w1 = std::vector<std::vector<double>>(input_dim, std::vector<double>(output_dim));
        b1 = static_cast<double>(rand()) / RAND_MAX - 0.5; 

        for (int i = 0; i < input_dim; ++i) {
            for (int j = 0; j < output_dim; ++j) {
                w1[i][j] = static_cast<double>(rand()) / RAND_MAX - 0.5; 
            }
        }
    }

    std::vector<double> feedforward(std::vector<double>& x) {
        std::vector<double> input_sum(w1[0].size(), b1);
        for (size_t i = 0; i < w1.size(); ++i) {
            for (size_t j = 0; j < w1[i].size(); ++j) {
                input_sum[j] += w1[i][j] * x[i];
            }
        }

        for (size_t i = 0; i < input_sum.size(); ++i) {
            input_sum[i] = sigmoid(input_sum[i]);
        }
        return input_sum;
    }

    void backpropagation(std::vector<double>& x, std::vector<double>& y, std::vector<double>& output) {
        std::vector<double> error(y.size());
        std::vector<double> delta(y.size());

        for (size_t i = 0; i < y.size(); ++i) {
            error[i] = y[i] - output[i];
            delta[i] = error[i] * deriv_sigmoid(output[i]);
        }

        for (size_t i = 0; i < w1.size(); ++i) {
            for (size_t j = 0; j < w1[i].size(); ++j) {
                w1[i][j] += learn_rate * delta[j] * x[i];
            }
        }

        b1 += learn_rate * delta[0];
    }
};

int main() {
    srand(time(0));

    int epochs = 10000; 
    NeuralNetwork nn(2, 2);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < x_data.size(); ++i) {
            std::vector<double> output = nn.feedforward(x_data[i]);
            nn.backpropagation(x_data[i], y_data[i], output);
        }

  
        if (epoch % 1000 == 0) {
            std::cout << "Epoch: " << epoch << std::endl;
            for (size_t i = 0; i < x_data.size(); ++i) {
                std::vector<double> output = nn.feedforward(x_data[i]);
                std::cout << "Output: " << output[0] << " " << output[1] << std::endl;
            }
        }
    }

    std::vector<double> test_input = { 180, 120 };
    std::vector<double> output = nn.feedforward(test_input);
    std::cout << "Final Output: " << output[0] << " " << output[1] << std::endl;

    return 0;
}
