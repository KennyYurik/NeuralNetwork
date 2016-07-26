#include "matrix.h"
#include <exception>
#include <chrono>
#include <numeric>

matrix::matrix(int n, int m, double fill_value) : N(n), M(m) {
	if (n <= 0 || m <= 0)
		throw std::exception("Bad size");
	data = vector<double>(N * M, fill_value);
}

matrix::matrix(int n, int m) : matrix(n, m, 0) {}

matrix::matrix(const vector<double>& column) {	
	if (column.empty())
		throw std::exception("Bad size");
	data = column;
	N = column.size();
	M = 1;
}

matrix::matrix(matrix&& m) {
	data = move(m.data);
	N = m.N;
	M = m.M;
}

int matrix::getN() { return N; }

int matrix::getM() { return M; }

double& matrix::at(int i, int j) {
	if (i < 0 || j < 0 || i >= N || j >= M)
		throw std::out_of_range("Bad index");
	return data.at(j * N + i);
}

double matrix::at(int i, int j) const{
	if (i < 0 || j < 0 || i >= N || j >= M)
		throw std::out_of_range("Bad index");
	return data.at(j * N + i);
}

void matrix::rand_initialize(double bound) {
	srand(clock());
	*this = apply([&](double x)-> double {
		return ((double)rand() / RAND_MAX) * 2 * bound - bound;
	});
}

matrix matrix::add_row(const vector<double>& row) {
	if (row.size() != M)
		throw std::exception("Bad size");
	matrix ans = *this;
	ans.N++;
	ans.data.insert(ans.data.end(), row.begin(), row.end());
	return ans;
}

matrix matrix::pop_row() {
	if (N < 2)
		throw std::exception("N < 2");
	matrix ans = *this;
	ans.N--;
	ans.data.erase(ans.data.end() - M);
	return ans;
}

matrix matrix::trans() {
	matrix ans(M, N);
	for (int i = 0; i < M; ++i)
		for (int j = 0; j < N; ++j)
			ans.at(i, j) = this->at(j, i);
	return ans;
}

matrix matrix::fill(double filler) { return matrix(N, M, filler); }

double matrix::sum() { return std::accumulate(data.begin(), data.end(), 0.0); }


matrix matrix::apply(std::function<double(double)> f) const {
	matrix ans = *this;
	for (auto &elem : ans.data) 
		elem = f(elem);
	return ans;
}

matrix& matrix::operator+=(const matrix& r) {
	if (N != r.N || M != r.M) {
		throw std::exception("Wrong size");
	}
	for (int i = 0; i < N * M; ++i) {
		data[i] = data[i] + r.data[i];
	}
	return *this;
}

matrix matrix::operator+(const matrix& r) const {
	matrix ans = *this;
	return ans += r;
}

matrix matrix::operator-() const { return apply([](double x) {return -x; }); }

matrix& matrix::operator-=(const matrix& r) { return *this += -r; }

matrix matrix::operator-(const matrix& r) const {
	matrix ans = *this;
	return ans -= r;
}

matrix& matrix::operator*=(const matrix& r) { return *this = *this * r; }

matrix matrix::operator*(const matrix& r) const {
	if (M != r.N) {
		throw std::exception("Bad size");
	}
	matrix ans(N, r.M);
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < r.M; ++j) {
			for (int l = 0; l < M; ++l) {
				ans.at(i, j) += this->at(i, l) * r.at(l, j);
			}
		}
	}
	return ans;
}	

matrix matrix::mult_by_elem(const matrix& r) const {
	if (N != r.N || M != r.M)
		throw std::exception("Bad size");
	matrix ans(*this);
	for (int i = 0; i < data.size(); ++i) {
		ans.data[i] *= r.data[i];
	}
	return ans;
}

matrix operator*(const matrix& l, double x) { return l.mult_by_elem(matrix(l).fill(x)); }
matrix operator*(double x, const matrix& r) { return r * x; }

matrix operator+(const matrix& l, double x) { return l + matrix(l).fill(x); }
matrix operator+(double x, const matrix& r) { return r + x; }

matrix operator-(const matrix& l, double x) { return l + (-x); }
matrix operator-(double x, const matrix& r) { return -r + x; }


