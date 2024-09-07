#include <iostream>
#include <vector>
#include <random>
#include "model.h"
#include "gnuplot-iostream.h"

// Implementations for MultiplyGate class
Eigen::MatrixXd MultiplyGate::forward(const Eigen::MatrixXd& W, const Eigen::MatrixXd& X) {
    return X * W;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> MultiplyGate::backward(const Eigen::MatrixXd& W, const Eigen::MatrixXd& X, const Eigen::MatrixXd& dZ) {
    Eigen::MatrixXd dW = X.transpose() * dZ;
    Eigen::MatrixXd dX = dZ * W.transpose();
    return { dW, dX };
}

// Implementations for AddGate class
Eigen::MatrixXd AddGate::forward(const Eigen::MatrixXd& X, const Eigen::MatrixXd& b) {
    return X.rowwise() + b.row(0);
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> AddGate::backward(const Eigen::MatrixXd& X, const Eigen::MatrixXd& b, const Eigen::MatrixXd& dZ) {
    Eigen::MatrixXd dX = dZ;
    Eigen::MatrixXd db = dZ.colwise().sum();
    return { db, dX };
}

// Implementations for Tanh class
Eigen::MatrixXd Tanh::forward(const Eigen::MatrixXd& X) {
    return X.array().tanh();
}

Eigen::MatrixXd Tanh::backward(const Eigen::MatrixXd& X, const Eigen::MatrixXd& top_diff) {
    Eigen::MatrixXd output = forward(X);
    return (1.0 - output.array().square()) * top_diff.array();
}

// Implementations for Softmax class
Eigen::MatrixXd Softmax::predict(const Eigen::MatrixXd& X) {
    Eigen::MatrixXd exp_scores = X.array().exp();
    return exp_scores.array().colwise() / exp_scores.rowwise().sum().array();
}

double Softmax::loss(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) {
    Eigen::Index num_examples = X.rows();
    Eigen::MatrixXd probs = predict(X);
    Eigen::VectorXd correct_logprobs(num_examples);
    for (Eigen::Index i = 0; i < num_examples; i++) {
        correct_logprobs(i) = -log(probs(i, y(i)));
    }
    double data_loss = correct_logprobs.sum();
    return data_loss / num_examples;
}

Eigen::MatrixXd Softmax::diff(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) {
    Eigen::Index num_examples = X.rows();
    Eigen::MatrixXd probs = predict(X);
    for (Eigen::Index i = 0; i < num_examples; i++) {
        probs(i, y(i)) -= 1;
    }
    return probs;
}

// Implementations for Model class
Model::Model(const std::vector<int>& layers_dim) {
    std::random_device rd;
    std::mt19937 gen(rd());
    for (size_t i = 0; i < layers_dim.size() - 1; i++) {
        std::normal_distribution<> d(0, 1.0 / sqrt(layers_dim[i]));
        W.push_back(Eigen::MatrixXd::NullaryExpr(layers_dim[i], layers_dim[i + 1], [&]() { return d(gen); }));
        b.push_back(Eigen::MatrixXd::NullaryExpr(1, layers_dim[i + 1], [&]() { return d(gen); }));
    }
}

double Model::calculate_loss(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) {
    MultiplyGate mulGate;
    AddGate addGate;
    Tanh layer;
    Softmax softmaxOutput;

    Eigen::MatrixXd input = X;
    for (size_t i = 0; i < W.size(); i++) {
        Eigen::MatrixXd mul = mulGate.forward(W[i], input);
        Eigen::MatrixXd add = addGate.forward(mul, b[i]);
        input = layer.forward(add);
    }

    return softmaxOutput.loss(input, y);
}

Eigen::VectorXi Model::predict(const Eigen::MatrixXd& X) {
    MultiplyGate mulGate;
    AddGate addGate;
    Tanh layer;
    Softmax softmaxOutput;

    Eigen::MatrixXd input = X;
    for (size_t i = 0; i < W.size(); i++) {
        Eigen::MatrixXd mul = mulGate.forward(W[i], input);
        Eigen::MatrixXd add = addGate.forward(mul, b[i]);
        input = layer.forward(add);
    }

    Eigen::MatrixXd probs = softmaxOutput.predict(input);
    return probs.rowwise().maxCoeff().cast<int>();
}

void Model::train(const Eigen::MatrixXd& X, const Eigen::VectorXi& y, int num_passes, double epsilon, double reg_lambda, bool print_loss) {
    MultiplyGate mulGate;
    AddGate addGate;
    Tanh layer;
    Softmax softmaxOutput;

    for (int epoch = 0; epoch < num_passes; epoch++) {
        Eigen::MatrixXd input = X;
        std::vector<std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>> forward;
        forward.emplace_back(Eigen::MatrixXd(), Eigen::MatrixXd(), input);

        for (size_t i = 0; i < W.size(); i++) {
            Eigen::MatrixXd mul = mulGate.forward(W[i], input);
            Eigen::MatrixXd add = addGate.forward(mul, b[i]);
            input = layer.forward(add);
            forward.emplace_back(mul, add, input);
        }

        Eigen::MatrixXd dtanh = softmaxOutput.diff(std::get<2>(forward.back()), y);
        for (Eigen::Index i = static_cast<Eigen::Index>(forward.size()) - 1; i > 0; i--) {
            Eigen::MatrixXd dadd = layer.backward(std::get<1>(forward[i]), dtanh);
            Eigen::MatrixXd db, dmul;
            std::tie(db, dmul) = addGate.backward(std::get<0>(forward[i]), b[i - 1], dadd);
            Eigen::MatrixXd dW, dX;
            std::tie(dW, dX) = mulGate.backward(W[i - 1], std::get<2>(forward[i - 1]), dmul);

            dW += reg_lambda * W[i - 1];  // Add regularization terms

            b[i - 1] -= epsilon * db;
            W[i - 1] -= epsilon * dW;

            dtanh = dX;
        }

        if (print_loss && epoch % 1000 == 0) {
            std::cout << "Loss after iteration " << epoch << ": " << calculate_loss(X, y) << std::endl;
        }
    }
}

void plot_decision_boundary(std::function<Eigen::VectorXi(const Eigen::MatrixXd&)> pred_func, const Eigen::MatrixXd& X, const Eigen::VectorXi& y) {
    double x_min = X.col(0).minCoeff() - 0.5;
    double x_max = X.col(0).maxCoeff() + 0.5;
    double y_min = X.col(1).minCoeff() - 0.5;
    double y_max = X.col(1).maxCoeff() + 0.5;
    double h = 0.01;

    Eigen::VectorXd xx = Eigen::VectorXd::LinSpaced(static_cast<Eigen::Index>((x_max - x_min) / h), x_min, x_max);
    Eigen::VectorXd yy = Eigen::VectorXd::LinSpaced(static_cast<Eigen::Index>((y_max - y_min) / h), y_min, y_max);
    Eigen::MatrixXd grid_x = xx.replicate(1, yy.size());
    Eigen::MatrixXd grid_y = yy.transpose().replicate(xx.size(), 1);

    Eigen::MatrixXd xy(xx.size() * yy.size(), 2);
    Eigen::Index k = 0;
    for (Eigen::Index i = 0; i < xx.size(); i++) {
        for (Eigen::Index j = 0; j < yy.size(); j++) {
            xy(k, 0) = xx(i);
            xy(k, 1) = yy(j);
            k++;
        }
    }

    Eigen::VectorXi Z = pred_func(xy);
    Eigen::MatrixXd Z_matrix(xx.size(), yy.size());
    for (Eigen::Index i = 0; i < xx.size(); i++) {
        for (Eigen::Index j = 0; j < yy.size(); j++) {
            Z_matrix(i, j) = Z(i * yy.size() + j);
        }
    }

    Gnuplot gp("\"D:/data/gnuplot.exe\"");
    gp << "set xrange [" << x_min << ":" << x_max << "]\n";
    gp << "set yrange [" << y_min << ":" << y_max << "]\n";
    gp << "set pm3d map\n";
    gp << "set colorbox\n";
    gp << "splot '-' matrix with image\n";
    gp.send1d(Z_matrix.transpose());

    gp << "set palette defined (0 'blue', 1 'red')\n";
    gp << "plot '-' with points pt 7 lc variable\n";
    for (Eigen::Index i = 0; i < X.rows(); i++) {
        gp << X(i, 0) << " " << X(i, 1) << " " << y(i) << "\n";
    }
    gp << "e\n";

    gp << "pause mouse close\n";
}

int main() {
    Eigen::MatrixXd X(200, 2);
    Eigen::VectorXi y(200);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 0.2);

    for (int i = 0; i < 100; i++) {
        double t = 2 * M_PI * i / 100.0;
        X(i, 0) = t * cos(t) + d(gen);
        X(i, 1) = t * sin(t) + d(gen);
        y(i) = 0;

        X(i + 100, 0) = -t * cos(t) + d(gen);
        X(i + 100, 1) = -t * sin(t) + d(gen);
        y(i + 100) = 1;
    }

    Model model({ 2, 3, 2 });
    model.train(X, y, 20000, 0.01, 0.01, true);

    plot_decision_boundary([&](const Eigen::MatrixXd& x) { return model.predict(x); }, X, y);

    return 0;
}
