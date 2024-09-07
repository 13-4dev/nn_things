#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>  // for std::iota
#include <iterator> // for std::distance

// Function definitions

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double mse_loss(const Eigen::VectorXd& y_pred, const Eigen::VectorXd& y_true) {
    return (y_pred - y_true).squaredNorm() / 2.0;
}

Eigen::VectorXd mse_derivative(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred) {
    return (y_pred - y_true) / static_cast<double>(y_pred.size());
}

double deriv_sigmoid(double x) {
    double fx = sigmoid(x);
    return fx * (1 - fx);
}

Eigen::VectorXd softmax(const Eigen::VectorXd& xs) {
    Eigen::VectorXd exp_xs = xs.array().exp();
    return exp_xs / exp_xs.sum();
}

double normalization(double minNum, double maxNum, double x) {
    return (x - minNum) / (maxNum - minNum);
}

double FindMaxEigenArray(const Eigen::MatrixXd& arr, int axis) {
    return arr.col(axis).maxCoeff();
}

double FindMinEigenArray(const Eigen::MatrixXd& arr, int axis) {
    return arr.col(axis).minCoeff();
}

Eigen::MatrixXd NormalizeArray(Eigen::MatrixXd arr, double MaxElem, int axis) {
    arr.col(axis) /= MaxElem;
    return arr;
}

// NeuralNetwork class

class NeuralNetwork {
public:
    NeuralNetwork(double learning_rate, int epochs, int size_input, int neuron_hidden, int size_output) :
        learn_rate(learning_rate), epoch(epochs), size_input(size_input),
        size_output(size_output), num_neuron_hidden(neuron_hidden) {

        std::cout << "--------------------------------------------------\n";
        std::cout << "Neural network: \n";
        std::cout << "--------------------------------------------------\n";
        std::cout << "Input: " << size_input << "\n";
        std::cout << "Hidden: " << num_neuron_hidden << "\n";
        std::cout << "Output: " << size_output << "\n";

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, 0.01);

        w1 = Eigen::MatrixXd::NullaryExpr(num_neuron_hidden, size_input, [&]() { return d(gen); });
        w2 = Eigen::MatrixXd::NullaryExpr(size_output, num_neuron_hidden, [&]() { return d(gen); });
        b1 = Eigen::VectorXd::Zero(num_neuron_hidden);
        b2 = Eigen::VectorXd::Zero(size_output);

        std::cout << "w1: " << w1 << "\n";
        std::cout << "w2: " << w2 << "\n";
        std::cout << "--------------------------------------------------\n";
    }

    Eigen::VectorXd feedforward(const Eigen::VectorXd& x) {
        input_data = x;
        z1 = (w1 * x) + b1;
        sigmoid_hidden = z1.unaryExpr(&sigmoid);
        z2 = (w2 * sigmoid_hidden) + b2;
        sigmoid_output = z2.unaryExpr(&sigmoid);
        return sigmoid_output;
    }

    void backpropagation(const Eigen::VectorXd& x, const Eigen::VectorXd& y, int i) {
        Eigen::VectorXd delta = mse_derivative(y, x).cwiseProduct(z2.unaryExpr(&deriv_sigmoid));

        Eigen::MatrixXd grad_w2 = delta * sigmoid_hidden.transpose();
        Eigen::VectorXd grad_b2 = delta;

        w2 -= learn_rate * grad_w2;
        b2 -= learn_rate * grad_b2;

        Eigen::VectorXd delta_input = (w2.transpose() * delta).cwiseProduct(z1.unaryExpr(&deriv_sigmoid));
        Eigen::MatrixXd grad_w1 = delta_input * input_data.transpose();
        Eigen::VectorXd grad_b1 = delta_input;

        w1 -= learn_rate * grad_w1;
        b1 -= learn_rate * grad_b1;
    }

    void train(const std::vector<Eigen::VectorXd>& x, const std::vector<Eigen::VectorXd>& y, const std::vector<Eigen::MatrixXd>& all_train) {
        int size_data = static_cast<int>(x.size());
        int batch_size = 40;
        int num_batch = static_cast<int>(std::round(size_data / static_cast<double>(batch_size)));

        for (int ep = 0; ep < epoch; ++ep) {
            // Shuffle data
            std::vector<int> indices(size_data);
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), std::mt19937{ std::random_device{}() });

            for (int index = 0; index < num_batch; ++index) {
                int stop = std::min(index + batch_size, size_data);

                for (int i = index; i < stop; ++i) {
                    int idx = indices[i];
                    Eigen::VectorXd pred = feedforward(x[idx]);
                    backpropagation(pred, y[idx], index);
                    double error = mse_loss(pred, y[idx]);

                    if (ep % 10 == 0 && i == stop - 1) {
                        std::cout << "--------------------\n";
                        std::cout << "epoch: " << ep << "\n";
                        std::cout << "error: " << error << "\n";
                        Eigen::VectorXd setosa_input(4);
                        setosa_input << 5.1, 3.5, 1.4, 0.2;
                        Eigen::VectorXd versicolor_input(4);
                        versicolor_input << 5.5, 2.5, 4.0, 1.3;
                        Eigen::VectorXd virginica_input(4);
                        virginica_input << 5.9, 3.0, 5.1, 1.8;
                        std::cout << "setosa: " << feedforward(setosa_input) << "\n";
                        std::cout << "setosa argmax: " << feedforward(setosa_input).maxCoeff() << "\n";
                        std::cout << "versicolor: " << feedforward(versicolor_input) << "\n";
                        std::cout << "versicolor argmax: " << feedforward(versicolor_input).maxCoeff() << "\n";
                        std::cout << "virginica: " << feedforward(virginica_input) << "\n";
                        std::cout << "virginica argmax: " << feedforward(virginica_input).maxCoeff() << "\n";
                    }
                }
            }
        }
    }

private:
    double learn_rate;
    int epoch;
    int size_input;
    int size_output;
    int num_neuron_hidden;
    Eigen::MatrixXd w1, w2;
    Eigen::VectorXd b1, b2;
    Eigen::VectorXd input_data, z1, z2, sigmoid_hidden, sigmoid_output;
};

// Main code

int main() {
    // Load Iris dataset (hardcoded for simplicity)
    Eigen::MatrixXd X(150, 4);
    Eigen::VectorXd y(150);

    // Initialize the neural network
    NeuralNetwork network(0.1, 100, 4, 5, 3);

    // Prepare training data
    std::vector<Eigen::VectorXd> X_data(150, Eigen::VectorXd(4));
    std::vector<Eigen::VectorXd> y_data(150, Eigen::VectorXd(3));
    std::vector<Eigen::MatrixXd> all_train;

    // Manually populate X_data and y_data with Iris dataset values...

    // Train the network
    network.train(X_data, y_data, all_train);

    return 0;
}
