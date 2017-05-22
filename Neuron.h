#include <iostream>
#include <armadillo>
#include <vector>
#include <map>

using namespace arma;
using namespace std;

class Neuron
{
public:
	Neuron():m_output(0){}
	Neuron(float v):m_output(v){}
	~Neuron(){}
	float m_output;
	
};


class Layer
{
public:
	Layer(int n,int f,int c);
	Layer();
	~Layer(){}
	std::vector<Neuron> listNeurons;
	Mat<double> matrixWeights;
	vec listOutput;
	vec listPropagation;
	vec listError;
	vec listDerivPropagation;
	
};

Layer::Layer()
{

}

Layer::Layer(int out,int f,int c)
{
	listOutput = zeros(out);
	//listPropagation = zeros(prop);
	//cambiar aleatorio
	//Mat<double> B = randu(3,3);
	matrixWeights.zeros(f,c); 
}