#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

class NN {
public:
    vector<double> p;
    function<double(const vector<double>&, const vector<double>&)> f;
    vector<NN*> children;
    vector<double> args;
    vector<double> g;

    NN(const vector<double>& p, function<double(const vector<double>&, const vector<double>&)> f, const vector<NN*>& children)
        : p(p), f(f), children(children) {}

    double operator()(const vector<double>& args, const vector<double>& shift = {}, vector<int> counter = { 0 }, bool update = false) {
        this->args.clear();
        for (auto& child : children) {
            if (child) {
                this->args.push_back((*child)(args, shift, counter, update));
            }
            else {
                this->args.push_back(0);
            }
        }

        for (int i = counter[0]; i < static_cast<int>(min(shift.size(), p.size())); ++i) {
            p[i] += shift[i];
        }

        double r = f(this->args, p);

        if (!update) {
            for (int i = counter[0]; i < static_cast<int>(min(shift.size(), p.size())); ++i) {
                p[i] -= shift[i];
            }
        }

        counter[0] += static_cast<int>(p.size());
        return r;
    }

    void grad(const vector<double>& args, vector<double>& gr, double delta = 1e-5, vector<int> counter = { 0 }, double c = 1.0) {
        double r = 0;
        this->args.clear();
        for (auto& child : children) {
            if (child) {
                this->args.push_back((*child)(args));
            }
            else {
                this->args.push_back(0);
            }
        }

        for (size_t i = 0; i < this->args.size(); ++i) {
            if (children[i]) {
                this->args[i] += delta;
                r = f(this->args, p);
                this->args[i] -= delta;
                this->args[i] -= delta;
                r -= f(this->args, p);
                this->args[i] += delta;
                r /= (2 * delta);
                r *= c;
                children[i]->grad(args, gr, delta, counter, r);
            }
        }

        for (size_t i = 0; i < p.size(); ++i) {
            p[i] += delta;
            r = f(this->args, p);
            p[i] -= delta;
            p[i] -= delta;
            r -= f(this->args, p);
            p[i] += delta;
            r /= (2 * delta);
            r *= c;

            if (counter[0] < static_cast<int>(gr.size())) {
                gr[counter[0]] = r;
            }
            else {
                gr.push_back(r);
            }
            counter[0]++;
        }
    }
};

class NNet {
public:
    struct Layer {
        function<double(double)> activation;
        int size;
        MatrixXd weights;
        VectorXd biases;
        VectorXd outputs;
        MatrixXd weight_grads;
        VectorXd bias_grads;
        VectorXd pre_activations;
    };

    vector<Layer> layers;

    NNet(const vector<pair<function<double(double)>, int>>& architecture) {
        int k = 0;
        for (const auto& l : architecture) {
            if (k == 0) {
                k = l.second;
            }
            layers.push_back({ l.first, l.second, MatrixXd::Zero(l.second, k), VectorXd::Zero(l.second),
                              VectorXd::Zero(l.second), MatrixXd::Zero(l.second, k),
                              VectorXd::Zero(l.second), VectorXd::Zero(l.second) });
            k = l.second;
        }
    }

    VectorXd operator()(const VectorXd& x) {
        if (layers.empty() || x.size() != layers[0].weights.cols()) {
            return x;
        }
        VectorXd y = x;
        for (auto& layer : layers) {
            y = layer.weights * y + layer.biases;
            layer.pre_activations = y;
            for (int i = 0; i < y.size(); ++i) {
                y[i] = layer.activation(y[i]);
            }
            layer.outputs = y;
        }
        return y;
    }

    double loss(const vector<pair<VectorXd, VectorXd>>& xy, function<double(double, double)> f) {
        double p = 0;
        for (const auto& [x, y] : xy) {
            if (layers.empty() || x.size() != layers[0].weights.cols() || y.size() != layers.back().biases.size() || !f) {
                continue;
            }
            VectorXd z = (*this)(x);
            for (int i = 0; i < y.size(); ++i) {
                p += f(z[i], y[i]);
            }
        }
        return p;
    }

    void grad(const vector<pair<VectorXd, VectorXd>>& xy, function<double(double, double)> f, double delta = 1e-3) {
        for (auto& layer : layers) {
            layer.weight_grads.setZero();
            layer.bias_grads.setZero();
        }

        for (const auto& [x, y] : xy) {
            if (layers.empty() || x.size() != layers[0].weights.cols() || y.size() != layers.back().biases.size() || !f) {
                continue;
            }

            VectorXd z = (*this)(x);
            for (int n = static_cast<int>(layers.size()) - 1; n >= 0; --n) {
                auto& layer = layers[n];
                VectorXd w = VectorXd::Zero(layer.pre_activations.size());
                if (n == layers.size() - 1) {
                    for (int i = 0; i < layer.pre_activations.size(); ++i) {
                        double p = z[i];
                        double q = (f(p + delta, y[i]) - f(p - delta, y[i])) / (2 * delta);
                        w[i] = q;
                    }
                }
                for (int i = 0; i < layer.pre_activations.size(); ++i) {
                    double p = layer.pre_activations[i];
                    double q = (layer.activation(p + delta) - layer.activation(p - delta)) / (2 * delta);
                    w[i] *= q;
                }
                VectorXd z_prev = (n > 0) ? layers[n - 1].outputs : x;
                for (int i = 0; i < layer.pre_activations.size(); ++i) {
                    layer.bias_grads[i] += w[i];
                    for (int j = 0; j < z_prev.size(); ++j) {
                        layer.weight_grads(i, j) += w[i] * z_prev[j];
                    }
                }
                if (n > 0) {
                    w = layer.weights.transpose() * w;
                }
            }
        }
    }

    pair<double, double> simple_descent(const vector<pair<VectorXd, VectorXd>>& xy, function<double(double, double)> f,
        double step = 1, double delta = 1e-3, int n = 8) {
        grad(xy, f, delta);
        double p = 0;
        double q = loss(xy, f);
        double r = 0;

        for (auto& layer : layers) {
            p += layer.weight_grads.squaredNorm();
            p += layer.bias_grads.squaredNorm();
        }
        r = p;
        if (p > 1e-16) {
            p = step / sqrt(p);
            for (auto& layer : layers) {
                layer.weight_grads *= p;
                layer.bias_grads *= p;
            }
        }

        for (int m = 0; m < n; ++m) {
            for (auto& layer : layers) {
                layer.weights -= layer.weight_grads;
                layer.biases -= layer.bias_grads;
            }
            p = loss(xy, f);
            if (p < q) {
                q = p;
                for (auto& layer : layers) {
                    layer.weight_grads *= 1.2;
                    layer.bias_grads *= 1.2;
                }
            }
            else {
                for (auto& layer : layers) {
                    layer.weights += layer.weight_grads;
                    layer.weight_grads *= -0.72;
                    layer.biases += layer.bias_grads;
                    layer.bias_grads *= -0.72;
                }
            }
        }
        return { q, r };
    }
};

int main() {
    int M = 20;
    auto s = [](double x) { return 1.0 / (exp(x) + 1.0); };
    auto l = [](double x, double y) { return 0.5 * pow(x - y, 2); };

    NNet n({ {s, M}, {s, M}, {s, M} });

    double delta = 1e-2;
    for (auto& layer : n.layers) {
        layer.biases = VectorXd::Random(M);
        layer.weights = MatrixXd::Random(M, M);
    }

    vector<pair<VectorXd, VectorXd>> xy;
    for (int i = 0; i < 200; ++i) {
        VectorXd x = VectorXd::Random(M);
        VectorXd y = VectorXd::Random(M);
        xy.emplace_back(x, y);
    }

    n.grad(xy, l);
    for (int m = 0; m < 2; ++m) {
        for (int i = 0; i < 2; ++i) {
            n.layers[m].biases[i] += delta;
            double p = n.loss(xy, l);
            n.layers[m].biases[i] -= 2 * delta;
            p -= n.loss(xy, l);
            p /= (2 * delta);
            n.layers[m].biases[i] += delta;
            cout << p << " " << n.layers[m].bias_grads[i] << endl;
            for (int j = 0; j < 2; ++j) {
                n.layers[m].weights(i, j) += delta;
                p = n.loss(xy, l);
                n.layers[m].weights(i, j) -= 2 * delta;
                p -= n.loss(xy, l);
                p /= (2 * delta);
                n.layers[m].weights(i, j) += delta;
                cout << p << " " << n.layers[m].weight_grads(i, j) << endl;
            }
        }
    }

    for (int i = 0; i < 200; ++i) {
        cout << n.simple_descent(xy, l, 1.0 / log(2 + i), 1e-3, 16).first << endl;
    }

    return 0;
}


