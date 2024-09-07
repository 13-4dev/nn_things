#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <vector>
#include <random>

using namespace Eigen;

// Функция градиентного спуска
std::pair<MatrixXd, std::vector<double>> GradientDescent(const MatrixXd& X, const MatrixXd& y, MatrixXd theta, double lr = 0.01, int n_iters = 100) {
    int m = y.rows();
    std::vector<double> costs;

    for (int i = 0; i < n_iters; ++i) {
        MatrixXd y_hat = X * theta;
        theta = theta - (lr / m) * (X.transpose() * (y_hat - y));
        double cost = (1.0 / (2 * m)) * (y_hat - y).array().square().sum();
        costs.push_back(cost);
    }

    return { theta, costs };
}

// Класс линейной регрессии
class LinearRegression {
public:
    double lr;
    int n_iters;
    std::vector<double> cost;
    MatrixXd theta;

    LinearRegression(double learning_rate = 0.01, int iterations = 1000)
        : lr(learning_rate), n_iters(iterations) {}

    void train(const MatrixXd& X, const MatrixXd& y) {
        theta = MatrixXd::Random(X.cols(), 1);
        std::tie(theta, cost) = GradientDescent(X, y, theta, lr, n_iters);
    }

    MatrixXd predict(const MatrixXd& X) const {
        return X * theta;
    }
};

int main() {
    // Генерация случайных данных
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 5);
    std::normal_distribution<> noise(0, 1);

    MatrixXd x(100, 1);
    MatrixXd y(100, 1);
    for (int i = 0; i < 100; ++i) {
        x(i, 0) = dis(gen);
        y(i, 0) = 2 + 3 * x(i, 0) + noise(gen);
    }

    // Добавление столбца единиц
    MatrixXd x_data(x.rows(), x.cols() + 1);
    x_data << MatrixXd::Ones(x.rows(), 1), x;

    std::cout << "Shape of x_data: " << x_data.rows() << "x" << x_data.cols() << std::endl;

    // Инициализация и обучение модели
    LinearRegression model(0.01, 1000);
    model.train(x_data, y);

    // Печать значений theta
    std::cout << "Thetas:\n" << model.theta << std::endl;

    // Предсказание
    MatrixXd y_predicted = model.predict(x_data);

    // Запись данных в файл для построения графика
    std::ofstream data_file("data.txt");
    for (int i = 0; i < x.rows(); ++i) {
        data_file << x(i, 0) << " " << y(i, 0) << " " << y_predicted(i, 0) << "\n";
    }
    data_file.close();

    // Построение графика через gnuplot
    std::system("gnuplot -persist -e \"set xlabel 'x'; set ylabel 'y'; plot 'data.txt' using 1:2 with points pointtype 7 pointsize 1 lc rgb 'blue', 'data.txt' using 1:3 with lines lc rgb 'red'\"");

    return 0;
}
