#include <fstream>
#include <string>
#include <algorithm>
#include "matrix.h"
#include "neural_network.h"


void main() {
	vector<std::pair<vector<double>, int>> dataset;
	for (int digit = 0; digit < 10; ++digit) {
		std::ifstream in("data/data" + std::to_string(digit), std::ios::binary);
		for (int i = 0; i < 1000; ++i) {
			vector<double> pixels;
			for (int j = 0; j < 28 * 28; ++j) {
				char c;
				in.read(&c, sizeof(c));
				pixels.push_back((double)c);
			}
			dataset.push_back({pixels, digit});
		}
	}
	std::random_shuffle(dataset.begin(), dataset.end());
	int sample_size = 0.8 * dataset.size();

	NeuralNetwork handwriting_digits{ 28 * 28, 100, 10 };
	
	for (int i = 0; i < sample_size; ++i)
		handwriting_digits.add_sample(dataset[i].first, dataset[i].second);

	//handwriting_digits.train();
	
	int bad_counter = 0;
	for (int i = sample_size; i < dataset.size(); ++i) {
		int result = handwriting_digits.get(dataset[i].first);
		if (result != dataset[i].second)
			bad_counter++;
	}

	std::ofstream out("out.txt");
	out << "samples size: " << sample_size << std::endl;
	out << "check size: " << dataset.size() - sample_size << std::endl;
	out << "correctness: " << (1 - bad_counter / ((double)dataset.size() - sample_size)) * 100 << "%" << std::endl;
}