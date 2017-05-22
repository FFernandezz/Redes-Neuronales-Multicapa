#include <iostream>
#include "NeuralNetwork.h"
#include "CImg.h"
#include <string>
#include <fstream>
#include <stdlib.h> 

using namespace arma;
using namespace std;
using namespace cimg_library;

#define nPixeles 784 

class TrainDigit
{
public:
	TrainDigit(){}
	~TrainDigit(){}
	vector<vec> listInput;
	vector<vec> listOutput;
	void saveFile(string filename);
	void print();
	
};

void TrainDigit::saveFile(string filename)
{
  string line;
  ifstream fileIn(filename);
  if (fileIn.is_open())
  {
    while ( getline (fileIn,line) )
    {	
      std::stringstream pixels(line);
      int index = 0;
      vec input = zeros(nPixeles);
  	  vec output = zeros(1);
	  for (std::string dato; std::getline(pixels,dato, ','); )
	  {
    	if(index==0)
    	{
    		output[0] = atoi(dato.c_str());
    		++index;
    	}
    	else
    	{
    		input[index] = atoi(dato.c_str());
    		++index; 
    	}	

	  }
	  listInput.push_back(input);
      listOutput.push_back(output);
    }
    fileIn.close();
  }

  else cout << "No se encuentra el archivo..."<<endl; 
}

void TrainDigit::print()
{
	cout<<"INPUT"<<endl;
	for(int i=0;i<listInput.size();++i)
	{
		for(int j=0;j<listInput[i].n_elem;++j)
		{
			cout<<listInput[i][j]<<" ";
		}
		cout<<endl;
	}
	cout<<"OUTPUT"<<endl;
	for(int i=0;i<listOutput.size();++i)
	{
		for(int j=0;j<listOutput[i].n_elem;++j)
		{
			cout<<listOutput[i][j]<<" ";
		}
		cout<<endl;
	}
		
}