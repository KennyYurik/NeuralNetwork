#include "neural_network.h"

NeuralNetwork::NeuralNetwork(std::initializer_list<int> list) : NeuralNetwork(vector<int>(list)) {}

NeuralNetwork::NeuralNetwork(vector<int>& layers) {
	if (layers.size() < 2)
		throw std::exception("Number of layers should be >= 2");
	for (int l_size : layers) {
		if (l_size <= 0)
			throw std::exception("Layer size should be > 0");
		this->layers.push_back(l_size);
		delta_small.push_back(matrix(l_size, 1));
		activations.push_back(matrix(l_size, 1));
	}
	for (int i = 1; i < this->layers.size(); ++i) {
		theta.push_back(matrix(this->layers[i], this->layers[i - 1] + 1));
	}
	for (auto& matrix : theta) {
		delta_big.push_back(matrix);
		matrix.rand_initialize(0.01);
	}
}

void NeuralNetwork::add_sample(vector<double> &x, int y) {
	if (x.size() != layers[0])
		throw std::exception("Wrong feature count");
	if (y < 0 || y >= layers.back())
		throw std::exception("bad Y");
	dataset.push_back({ x, y });
}

double NeuralNetwork::_cost(){
	if (dataset.empty())
		return 0;
	double J = 0;
	matrix zero(layers.back(), 1, 0);
	for (auto& sample : dataset) {
		matrix h = _feedforward(sample.x);
		matrix y = zero;
		y.at(sample.y, 0) = 1;
		auto res = -y.mult_by_elem(log(h)) - (1 - y).mult_by_elem(log(1 - h));
		J += res.sum();
	}
	for (auto& matrix : theta) {
		//regularized thetas, excluding bias coeffs
		J += (lambda / 2) * sqr(matrix.trans().pop_row()).sum();
	}
	J /= dataset.size();
	return J;
}

void NeuralNetwork::train() {
	if (dataset.empty()) {
		throw std::exception("no data to train");
	}
	for (int iter = 0; iter < 50; ++iter) {
		for (auto& m : delta_big) {
			m.fill(0);
		}
		matrix zero(layers.back(), 1, 0);
		for (auto& sample : dataset) {
			_feedforward(sample.x);
			matrix y = zero;
			y.at(sample.y, 0) = 1;
			delta_small.back() = activations.back() - y;
			for (int i = delta_small.size() - 2; i > 0; --i) {
				delta_small[i] = (sigrad((theta[i].trans() * delta_small[i + 1]).mult_by_elem(activations[i].add_row({ 1.0 })))).pop_row();
			}
			for (int i = 0; i < delta_big.size(); ++i) {
				delta_big[i] = delta_big[i] + delta_small[i + 1] * activations[i].add_row({ 1.0 }).trans();
			}
		}
		for (int i = 0; i < delta_big.size(); ++i) {
			delta_big[i] = delta_big[i] * (1.0 / dataset.size()) + theta[i].pop_row().add_row(vector<double>(theta[i].getM(), 0)) * (lambda / dataset.size());
		}
		vector<matrix> delta_test = theta;
		for (int k = 0; k < theta.size(); ++k) {
			matrix m = theta[k];
			m = m.fill(0);
			for (int i = 0; i < m.getN(); ++i) {
				for (int j = 0; j < m.getM(); ++j) {
					theta[k].at(i, j) += eps;
					double costplus = _cost();
					theta[k].at(i, j) -= 2 * eps;
					double costminus = _cost();
					theta[k].at(i, j) += eps;
					m.at(i, j) = (costplus - costminus) / (2 * eps);
				}
			}
			delta_test[k] = m;
		}
		for (int i = 0; i < theta.size(); ++i) {
			theta[i] = theta[i] - delta_test[i] * 0.01;
		}
	}
}

matrix NeuralNetwork::_feedforward(vector<double>& x) {
	if (x.size() != layers[0])
		throw std::exception("Wrong features count");
	matrix column(x);
	activations[0] = column;
	column = column.add_row({ 1.0 });
	for (int i = 0; i < theta.size(); ++i) {
		column = theta[i] * column;
		activations[i + 1] = column;
		column = sigmoid(column).add_row({ 1.0 });
	}
	return column.pop_row();
}

int NeuralNetwork::get(vector<double>& x) {
	auto column = _feedforward(x);
	double ans = -1;
	int index = 0;
	for (int i = 0; i < column.getN(); ++i) {
		if (ans < column.at(i, 0)) {
			ans = column.at(i, 0);
			index = i;
		}
	}
	return index;
}