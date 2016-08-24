#pragma once
#include "matrix.h"

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
	
	double _cost();

	matrix _feedforward(vector<double>& x);

public:
	// constructs network from layers' sizes
	NeuralNetwork(std::initializer_list<int> list);

	// constructs network from layers' sizes
	NeuralNetwork(vector<int>& layers);

	// add data sample to train
	void add_sample(vector<double> &x, int y);

	// construct thetas from samples
	void train();

	// feedforward data and get result group
	int get(vector<double> &x);
};