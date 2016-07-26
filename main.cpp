#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <tuple>
#include <numeric>
#include <functional>
//#include <cmath>
#include <chrono>
#include <algorithm>
#include "matrix.h"

using namespace std;
ofstream out("out.txt");



class NeuralNetwork {
private:
	struct Sample{
		vector<double> x;
		int y;
	}; 

	double lambda = 1;
	double eps = 0.0001;
	vector<int> layers;
	vector<matrix> theta;
	vector<Sample> dataset;
	vector<matrix> delta_small;
	vector<matrix> activations;
	vector<matrix> delta_big;

	
public:
	NeuralNetwork(initializer_list<int> list) : NeuralNetwork(vector<int>(list)) {
	}
		 
	NeuralNetwork(vector<int> layers) {
		if (layers.size() < 2)
			throw exception("Number of layers should be >= 2");
		for (int l_size : layers) {
			if (l_size <= 0)
				throw exception("Layer size should be > 0");
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

	void add_example(vector<double> &x, int y) {
		if (x.size() != layers[0])
			throw exception("Wrong feature count");
		if (y < 0 || y >= layers.back())
			throw exception("bad Y");
		dataset.push_back({x, y});
	}

	double cost(){
		if (dataset.empty())
			return 0;
		double J = 0;
		matrix zero(layers.back(), 1, 0);
		for (auto& sample : dataset) {
			matrix h = feedforward(sample.x);
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

	void train() {
		if (dataset.empty()) {
			throw exception("no data to train");
		}
		for (int iter = 0; iter < 50; ++iter) {
			out << iter << ". " << cost() << endl;
			for (auto& m : delta_big) {
				m.fill(0);
			}
			matrix zero(layers.back(), 1, 0);
			for (auto& sample : dataset) {
				feedforward(sample.x);
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
						double costplus = cost();
						theta[k].at(i, j) -= 2 * eps;
						double costminus = cost();
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

	matrix feedforward(vector<double> x) {
		if (x.size() != layers[0])
			throw exception("Wrong features count");
		matrix column(x);
		activations[0] = column;
		column = column.add_row({1.0});
		for (int i = 0; i < theta.size(); ++i) {
			column = theta[i] * column;
			activations[i + 1] = column;
			column = sigmoid(column).add_row({1.0});
		}
		return column.pop_row();
	}
};

void main() {
	NeuralNetwork test {2, 2};
	test.add_example(vector<double>({ 1.0, 222.0 }), 1);
	test.add_example(vector<double>({ -1.0, 12.0 }), 1);
	test.add_example(vector<double>({ 0.0, 4.0 }), 1);
	test.add_example(vector<double>({ 5.2, 0.5 }), 0);
	test.add_example(vector<double>({ 800.0, 2.7 }), 0);

	test.train();
	matrix m(1, 2);
	//m.apply(sqrt);
	//out << a;
	/*ifstream in("data0", ios::binary);
	ofstream out("out.txt");
	for (int i = 0; i < 1000; ++i) {
		for (int j = 0; j < 28; ++j) {
			for (int k = 0; k < 28; ++k) {
				char c;
				in.read(&c, sizeof(c));
				out << (c < 0 ? 0 : 1) << " ";
			}
			out << endl;
		}
		out << endl;
	}*/
}