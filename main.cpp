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

using namespace std;

class Matrix {
private:
	vector<double> data;
	int N, M;

public:
	
	// default zero N x M matrix
	Matrix(int n, int m) : N(n), M(m) {
		if (N <= 0 || M <= 0) 
			throw exception("Size should be positive.");
		data = vector<double>(N * M, 0);
	}

	// N x 1 column matrix from data
	Matrix(vector<double>& column) {
		data = column;
		N = column.size();
		M = 1;
	}

	Matrix(int n, int m, double fill_value) : Matrix(n, m) {
		data = vector<double>(N * M, fill_value);
	}

	// rvalue operator
	Matrix(Matrix&& m) {
		data = move(m.data);
		N = m.N;
		M = m.M;
	}

	int getN() {
		return N;
	}

	int getM() {
		return M;
	}

	// get/set element
	double& at(int i, int j) {
		if (i < 0 || j < 0 || i >= N || j >= M)
			throw out_of_range("Bad index");
		return data.at(j * N + i);
	}

	// initialize all elements with random on (-bound, bound)
	void rand_initialize(double bound) {
		srand(clock());
		*this = this->apply([&](double x)-> double {
			return ((double)rand() / RAND_MAX) * 2 * bound - bound;
		});
	}

	// increase matrix size from N x M to (N + 1) x M filled with new data
	Matrix add_row(vector<double> row) {
		if (row.size() != M)
			throw exception("Wrong size");
		Matrix ans = *this;
		ans.N++;
		ans.data.insert(ans.data.end(), row.begin(), row.end());
		return ans;
	}

	// add
	Matrix operator+(Matrix& r) {
		if (N != r.N || M != r.M) {
			throw exception("Wrong size");
		}
		Matrix ans = Matrix(N, M);
		for (int i = 0; i < N * M; ++i) {
			ans.data[i] = data[i] + r.data[i];
		}
		return ans;
	}

	// transpose
	Matrix trans() {
		Matrix ans(M, N);
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) {
				ans.at(i, j) = this->at(j, i);
			}
		}
		return ans;
	}

	// delete last row, N x M -> (N - 1) x M
	Matrix pop_row() {
		if (N < 2)
			throw exception("N < 2");
		Matrix ans = *this;
		ans.N--;
		ans.data.erase(ans.data.end() - M);
		return ans;
	}

	// multiply
	Matrix operator*(Matrix& r) {
		if (M != r.N) {
			throw exception("Wrong size");
		}
		Matrix ans(N, r.M);
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < r.M; ++j) {
				for (int l = 0; l < M; ++l) {
					ans.at(i, j) += this->at(i, l) * r.at(l, j);
				}
			}
		}
		return ans;
	}

	Matrix operator-(){
		Matrix ans = *this;
		for (auto& elem : ans.data) {
			elem = -elem;
		}
		return ans;
	}

	Matrix operator-(Matrix& r) {
		return *this + (-r);
	}

	// fold all elems
	double sum() {
		return accumulate(data.begin(), data.end(), 0.0);
	}

	// multiplication by elements
	Matrix mult(Matrix& r) {
		if (N != r.N || M != r.M)
			throw exception("wrong size");
		Matrix ans(*this);
		for (int i = 0; i < data.size(); ++i) {
			ans.data[i] *= r.data[i];
		}
		return ans;
	}

	// apply function to all elems
	Matrix apply(function<double(double)> f) {
		Matrix ans = *this;
		for (int i = 0; i < N * M; ++i) {
			ans.data[i] = f(ans.data[i]);
		}
		return ans;
	}

	friend class NeuralNetwork;
};

namespace funcs {
	double log(double x) {
		return log2(x);
	}

	double sigmoid(double x) {
		return 1 / (1 + exp(x)); 
	}

	function<double(double)> plus(double a) {
		return [a](double x) {return x + a; };
	}

	double sqr(double x) {
		return x * x;
	}
}

class NeuralNetwork {
private:
	struct Sample{
		vector<double> x;
		int y;
	}; 

	double lambda = 1;
	vector<int> layers;
	vector<Matrix> theta;
	vector<Sample> dataset;
	
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
		}
		for (int i = 1; i < this->layers.size(); ++i) {
			theta.push_back(Matrix(this->layers[i], this->layers[i - 1] + 1));
		}
		for (auto& matrix : theta) {
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
		Matrix zero(layers.back(), 1, 0);
		for (auto& sample : dataset) {
			Matrix h = feedforward(sample.x);
			Matrix y = zero;
			y.at(sample.y, 0) = 1;
			// -y * log(h) - (1 - y) * log(1 - h)
			J += (-y.mult(h.apply(funcs::log)) - ((-y).apply(funcs::plus(1)).mult((-h).apply(funcs::plus(1))))).sum();
		}
		for (auto& matrix : theta) {
			//regularized thetas, excluding bias coeffs
			J += lambda / 2 * (matrix.trans().pop_row().apply(funcs::sqr)).sum();
		}
		J /= dataset.size();
		return J;
	}

	void train() {

	}

	Matrix feedforward(vector<double> x) {
		if (x.size() != layers[0])
			throw exception("Wrong features count");
		Matrix column(x);
		column = column.add_row({1.0});
		for (auto& matrix : theta) {
			column = matrix * column;
			column = column.apply(funcs::sigmoid).add_row({1.0});
		}
		return column.pop_row();
	}
};

void main() {
	NeuralNetwork test {1, 2};
	test.add_example(vector<double>(1, 1), 0);
	double cost = test.cost();
	ofstream out("out.txt");
	Matrix m(1, 2);
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