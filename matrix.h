#pragma once;

#include <cmath>
#include <vector>
#include <functional>

using std::vector;

class matrix {
private:
	vector<double> data;
	int N, M;

public:

	// default zero N x M matrix
	matrix(int n, int m);

	// N x 1 column matrix from data
	matrix(const vector<double>& column);

	// N x M matrix where all elements are equal to fill_value
	matrix(int n, int m, double fill_value);

	// rvalue operator
	matrix(matrix&& m);

	// row counter
	int getN();

	// column counter
	int getM();

	// get/set element
	double& at(int i, int j);
	double at(int i, int j) const;
	// initialize all elements with random values on (-bound, bound)
	void rand_initialize(double bound);

	// increase matrix size from N x M to (N + 1) x M filled with new data
	matrix add_row(const vector<double>& row);
	
	// delete last row, N x M -> (N - 1) x M
	matrix pop_row();

	// transpose
	matrix trans();

	// fill with value = filler
	matrix fill(double filler);
	
	
	// fold all elems
	double sum();

	// apply function to all elems
	matrix apply(std::function<double(double)> f) const;

	// OPERATIONS

	// add
	matrix& operator+=(const matrix& r);
	matrix operator+(const matrix& r) const;
	
	// sub
	matrix operator-() const;
	matrix operator-(const matrix& r) const;
	matrix& operator-=(const matrix& r);

	// multiply
	matrix operator*(const matrix& r) const;
	matrix& operator*=(const matrix& r);

	// multiply by elements
	matrix mult_by_elem(const matrix& r) const;

	// operations with matrix and number
	friend matrix operator*(const matrix& l, double x);
	friend matrix operator*(double x, const matrix& r);

	friend matrix operator+(const matrix& l, double x);
	friend matrix operator+(double x, const matrix& r);

	friend matrix operator-(const matrix& l, double x);
	friend matrix operator-(double x, const matrix& r);
};



matrix log(matrix& m) {
	return m.apply([](double x) { return std::log(x); });
}

matrix sigmoid(matrix& m) {
	return m.apply([](double x) { return 1 / (1 + exp(-x)); });
}

matrix sigrad(matrix& m) {
	return m.apply([](double z) { return z * (1 - z); });
}

matrix sqr(matrix& m) {
	return m.apply([](double x) {return x * x; });
}