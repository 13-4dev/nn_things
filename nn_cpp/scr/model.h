#pragma once
#ifndef MODEL_H
#define MODEL_H

#include <Eigen/Dense>
#include <vector>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class MultiplyGate {
public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd& W, const Eigen::MatrixXd& X);
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> backward(const Eigen::MatrixXd& W, const Eigen::MatrixXd& X, const Eigen::MatrixXd& dZ);
};

class AddGate {
public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd& X, const Eigen::MatrixXd& b);
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> backward(const Eigen::MatrixXd& X, const Eigen::MatrixXd& b, const Eigen::MatrixXd& dZ);
};

class Tanh {
public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd& X);
    Eigen::MatrixXd backward(const Eigen::MatrixXd& X, const Eigen::MatrixXd& top_diff);
};

class Softmax {
public:
    Eigen::MatrixXd predict(const Eigen::MatrixXd& X);
    double loss(const Eigen::MatrixXd& X, const Eigen::VectorXi& y);
    Eigen::MatrixXd diff(const Eigen::MatrixXd& X, const Eigen::VectorXi& y);
};

class Model {
public:
    std::vector<Eigen::MatrixXd> b;
    std::vector<Eigen::MatrixXd> W;

    Model(const std::vector<int>& layers_dim);

    double calculate_loss(const Eigen::MatrixXd& X, const Eigen::VectorXi& y);
    Eigen::VectorXi predict(const Eigen::MatrixXd& X);
    void train(const Eigen::MatrixXd& X, const Eigen::VectorXi& y, int num_passes = 20000, double epsilon = 0.01, double reg_lambda = 0.01, bool print_loss = false);
};

#endif // MODEL_H
