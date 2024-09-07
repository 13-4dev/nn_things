#include <iostream>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

class NeuralNetwork {
private:
    vector<MatrixXd> weights;
    vector<VectorXd> biases;
    vector<VectorXd> z;
    vector<VectorXd> a;
    double alpha;
    int iteration;
    double error;
    VectorXd derror;

    VectorXd dtanh(const VectorXd& x) {
        return 1.0 - x.array().tanh().square();
    }

public:
    NeuralNetwork(const vector<int>& units_per_layer_list, int iter, double learn_rate) {
        alpha = learn_rate;
        iteration = iter;
        int l = units_per_layer_list.size();

        for (int i = 0; i < l - 1; i++) {
            weights.push_back(MatrixXd::Random(units_per_layer_list[i + 1], units_per_layer_list[i]));
            biases.push_back(VectorXd::Random(units_per_layer_list[i + 1]));
        }
    }

    void show_weights() {
        for (const auto& w : weights) {
            cout << w << endl << endl;
        }
    }

    VectorXd forward(const VectorXd& input, double label) {
        if (input.size() != weights[0].cols()) {
            throw runtime_error("Invalid input size!");
        }

        VectorXd output = input;
        a.push_back(output); 
        for (size_t i = 0; i < weights.size(); ++i) {
            VectorXd z_temp = weights[i] * output + biases[i];
            z.push_back(z_temp);
            output = z_temp.array().tanh();
            a.push_back(output);
        }

        error = 0.5 * pow((output(0) - label), 2);
        derror = output - VectorXd::Constant(output.size(), label);
        return output;
    }

    void backpropagate() {
        VectorXd delta = derror.cwiseProduct(dtanh(z.back()));
        weights.back() -= alpha * delta * a[a.size() - 2].transpose();
        biases.back() -= alpha * delta;

        for (int i = weights.size() - 2; i >= 0; --i) {
            delta = (weights[i + 1].transpose() * delta).cwiseProduct(dtanh(z[i]));
            weights[i] -= alpha * delta * a[i].transpose();
            biases[i] -= alpha * delta;
        }
    }

    void train(const vector<VectorXd>& input_data, const vector<double>& input_labels) {
        int n = input_data.size();

        for (int i = 0; i < iteration; ++i) {
            double average_error = 0.0;

            for (size_t j = 0; j < n; ++j) {
                forward(input_data[j], input_labels[j]);
                average_error += error;
                backpropagate();

                a.clear();
                z.clear();
            }

            cout << "Iteration #" << i + 1 << " Error: " << average_error / n << endl;
        }
    }

    void predict(const VectorXd& input_data) {
        cout << forward(input_data, 0.0) << endl;
        error = 0;
        a.clear();
        z.clear();
    }
};

int main() {
    vector<int> layers = { 2, 3, 2, 1 };
    NeuralNetwork nn(layers, 6, 0.5);

    vector<VectorXd> input_data = { VectorXd(2) };
    input_data[0] << 1.0, 2.0;

    vector<double> input_labels = { 0.5 };

    nn.train(input_data, input_labels);
    nn.predict(VectorXd::Constant(2, 1.0));

    return 0;
}
