#include <iostream>
#include "Neuron.h"
#include <map>
#include <math.h> 

using namespace std;
using namespace arma;

class NeuralNetwork
{
public:
	NeuralNetwork(){}
	NeuralNetwork(int nOcultas, vector<int> listNOcultas, vec input, vec output);
	~NeuralNetwork(){}
	vector<Layer> m_NNetwork;
	std::vector<int> setNLayers(int n, std::vector<int> listHidden);
	void printValues();
	void forward();
	void backPropagation();
	void updateInputOutput(vec input,vec output);
	float getError(vec d,vec output);
	NeuralNetwork Train(vector<vec > listTrain, vector<vec > listOutput,int nOcultas, vector<int> listNOcultas);
	void Test(vector<vec > listTest,vector<vec > listTrain, vector<vec > listOutput,int nOcultas, vector<int> listNOcultas);

	int m_nlayersHidden;
	int m_nlayers;
	vector<int> m_listNLayerHidden;
	std::vector<int> m_listNLayers;
	//std::map< int, std::map<int, float> > matrixTest;
	vec m_listInput;
	vec m_listOutput;
	float taza_Aprendizaje;

	float actFunction(float a);
	float derActFunction(float a);
};

std::vector<int> NeuralNetwork::setNLayers(int n, std::vector<int> listHidden)
{
	std::vector<int> listNLayer;
	auto it = listNLayer.begin();
	listNLayer.insert(it,m_listInput.n_elem);
	for(int i=1;i<=n;++i)
	{
		auto iterator = listNLayer.begin();
		listNLayer.insert(iterator+i,listHidden[i-1]);
	}
	listNLayer.push_back(m_listOutput.n_elem);
	return listNLayer;
}


float NeuralNetwork::actFunction(float a){return (1.0/(1.0+exp(-a)));}

float NeuralNetwork::derActFunction(float a){return exp(-a)/pow((1.0+exp(-a)),2);}

NeuralNetwork::NeuralNetwork(int nOcultas, vector<int> listNOcultas, vec input, vec output)
{
	m_nlayersHidden=nOcultas;
	m_listNLayerHidden = listNOcultas;

	m_nlayers = m_nlayersHidden +2;
	//m_listInput = {1,2.5};29
	m_listInput = input;
	m_listOutput = output;
	taza_Aprendizaje = 0.5;

	m_listNLayers = setNLayers(m_nlayersHidden,m_listNLayerHidden);

	//set up NN
	for(int i=0;i<m_listNLayers.size();++i)
	{
		if (i==0)
		{
			m_NNetwork.push_back(Layer(m_listNLayers[i]+1,0,0));
			m_NNetwork[i].listNeurons.push_back(Neuron(1));
			m_NNetwork[i].listOutput(0)=1;

			for(int j=0;j<m_listNLayers[i];++j)
			{
				m_NNetwork[i].listNeurons.push_back(Neuron(m_listInput[j]));
				m_NNetwork[i].listOutput(j+1)=m_listInput[j];
			}
		}
		else if(i==m_listNLayers.size()-1)
		{
			m_NNetwork.push_back(Layer(m_listNLayers[i],m_listNLayers[i-1]+1,m_listNLayers[i]));
			for(int j=0;j<m_listNLayers[i];++j)
			{
				m_NNetwork[i].listNeurons.push_back(Neuron());
			}
		}
		else
		{
			m_NNetwork.push_back(Layer(m_listNLayers[i]+1,m_listNLayers[i-1]+1,m_listNLayers[i]));
			m_NNetwork[i].listNeurons.push_back(Neuron(1));
			m_NNetwork[i].listOutput(0)=1;
			for(int j=0;j<m_listNLayers[i];++j)
				m_NNetwork[i].listNeurons.push_back(Neuron());
		}	
	}
}


void NeuralNetwork::forward()
{
	for(int i=0;i<m_NNetwork.size();++i)
	{
		if (i>0)
		{
			m_NNetwork[i].listPropagation = m_NNetwork[i].matrixWeights.t()*m_NNetwork[i-1].listOutput;
			if (i==m_NNetwork.size()-1)
			{
				for(int j=0;j<m_NNetwork[i].listPropagation.n_elem;++j)
				{
					m_NNetwork[i].listOutput(j)=actFunction(m_NNetwork[i].listPropagation[j]);
				}
			}
			else
			{
				for(int j=0;j<m_NNetwork[i].listPropagation.n_elem;++j)
				{
					m_NNetwork[i].listOutput(j+1)=actFunction(m_NNetwork[i].listPropagation[j]);
				}
			}
			

		}
	}
}

void NeuralNetwork::backPropagation()
{
	for(int i=m_NNetwork.size()-1;i>=0;--i)
	{
		m_NNetwork[i].listDerivPropagation = zeros(m_NNetwork[i].listPropagation.n_elem);
		for (int j=0;j<m_NNetwork[i].listPropagation.n_elem;++j)
			m_NNetwork[i].listDerivPropagation(j) = derActFunction(m_NNetwork[i].listPropagation[j]);

		if(i==m_NNetwork.size()-1)
		{
			//calcular el error, capa salida
			m_NNetwork[i].listError = (m_listOutput - m_NNetwork[i].listOutput) * m_NNetwork[i].listDerivPropagation;
			//actualizar los pesos
			m_NNetwork[i].matrixWeights +=  0.5* (m_NNetwork[i-1].listOutput*m_NNetwork[i].listError.t());
	

		}
		else if (i>0)
		{
			//calcular el error, capa oculta
			int filas = m_NNetwork[i+1].matrixWeights.n_rows;
			int columnas = m_NNetwork[i+1].matrixWeights.n_cols;
			m_NNetwork[i].listError = m_NNetwork[i+1].matrixWeights.submat(1,0,filas-1,columnas-1)*m_NNetwork[i+1].listError%m_NNetwork[i].listDerivPropagation;
			m_NNetwork[i].matrixWeights +=  0.5* (m_NNetwork[i-1].listOutput*m_NNetwork[i].listError.t());

		}
	}
}

void NeuralNetwork::updateInputOutput(vec input,vec output)
{
	vec newInput = zeros(input.n_elem +1);
	newInput[0] = 1;
	for(int j=1;j<=input.n_elem;++j){newInput[j]=input[j-1]; }
	m_NNetwork[0].listOutput = newInput;
	m_listOutput = output;
}

float NeuralNetwork::getError(vec d,vec output)
{
	float total = 0;
	for(int i=0;i<d.n_elem;++i)
	{
		total += pow((d[i] - output[i]),2);
	}
	return total;
}

NeuralNetwork NeuralNetwork::Train(vector<vec > listTrain, vector<vec > listOutput,int nOcultas, vector<int> listNOcultas)
{
	cout<<"TRAINING"<<endl;
	int ToleError = 0.01;
	NeuralNetwork NN(nOcultas,listNOcultas,listTrain[0],listOutput[0]);
	for(int i=1;i<listTrain.size();++i)
	{
		NN.forward();
		float error = getError(NN.m_NNetwork[NN.m_nlayers-1].listOutput,NN.m_listOutput);
		if (error <= ToleError)
		{
			NN.updateInputOutput(listTrain[i],listOutput[i]);
		}
		else
		{
			NN.backPropagation();
			NN.updateInputOutput(listTrain[i],listOutput[i]);
		}

	}
	return NN;
}

void NeuralNetwork::Test(vector<vec > listTest,vector<vec > listTrain, vector<vec > listOutput,int nOcultas, vector<int> listNOcultas)
{
	NeuralNetwork NN = Train(listTrain,listOutput,nOcultas,listNOcultas);
	cout<<"TEST"<<endl;
	for(int i=0;i<listTest.size();++i)
	{
		
		vec newInput = zeros(listTest[i].n_elem +1);
		newInput[0] = 1;
		for(int j=1;j<=listTest[i].n_elem;++j){newInput[j]=listTest[i][j-1]; }
		NN.m_NNetwork[0].listOutput = newInput;
		NN.forward();

		cout<<"salida"<<endl;
		cout<<NN.m_NNetwork[NN.m_nlayers-1].listOutput<<endl;
	}

}

void NeuralNetwork::printValues()
{
	for(int i=0;i<m_NNetwork.size();++i)
	{
			cout<<"Layer: "<<i<<endl;
			std::cout << "Lista Propagacion:\n" << m_NNetwork[i].listPropagation << "\n";
			std::cout << "Lista Output:\n" << m_NNetwork[i].listOutput << "\n";
			std::cout << "Weight:\n" << m_NNetwork[i].matrixWeights << "\n";
	}
}




