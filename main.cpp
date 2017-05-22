#include <iostream>
#include "TrainDigit.h"

using namespace std;
using namespace arma;
using namespace cimg_library; 

int main()
{
	//function=4x^2 + 4y^2
	/*int nOcultas = 1;
	vector<int> listNOcultas = {2};
	std::vector<vec> input;
	std::vector<vec> output;
	std::vector<vec> test;
	input.push_back({0,2});
	input.push_back({5,4});
	input.push_back({6,7});
	input.push_back({3,1});
	input.push_back({10,4});

	output.push_back({8});
	output.push_back({164});
	output.push_back({340});
	output.push_back({16});
	output.push_back({464});

	test.push_back({1,2});
	test.push_back({2,3});
	test.push_back({5,6});
	test.push_back({7,8});

	NeuralNetwork NN;
	NN.Test(test,input,output,nOcultas,listNOcultas);*/

	//Image
	//compile g++ -Wall -o lena lena.cpp -lpthread -lX11
	/*CImg<unsigned char> img(640,400);
	img.fill(0);
	img(100,100) = 254;
	img.display("My first CImg code");*/
	TrainDigit TD;
	TD.saveFile("DataBase/mnist_test_10.csv");
	TD.print();


	return 0;
}