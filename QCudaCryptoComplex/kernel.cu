#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <string>
#include <fstream>
using namespace std;

#define SIZE 1280

__constant__ unsigned int gKey[8];
__constant__ short int gTable[8][16];

__device__ void F(unsigned int *itsR)
{
	unsigned int r[8];
	unsigned int mask = 0xF;
	for(int i = 0; i < 8; i ++) {
		r[i] = (*itsR & mask) >> 4 * i;
		mask <<= 4;
	}
	*itsR = 0;
	for(int i = 7; i >= 0; i --) {
		(*itsR) <<= 4;
		r[i] = gTable[i][r[i]];
		(*itsR) += r[i];
	}
	
}

__global__ void gostEncrypt(unsigned long long *data, unsigned long long *result) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int L = ((data[index] & 0xFFFFFFFF00000000) >> 32);
	unsigned int R = (data[index] & 0x00000000FFFFFFFF);
	const long long number = 4294967296L;
	int j = 0;
	for(int i = 1; i <= 32; i ++) {
	    unsigned int V = R;
	    if(i < 25)
			j = (i - 1) % 8;
		else
			j = (32 - i) % 8;
		long long buf = R + gKey[j];
		while(buf >= number)
			buf = buf - number;
		R = buf;
		F(&R);
		unsigned long long mask = 0x80000000;
		unsigned int leftBit;
		for(int k = 0; k < 11; k ++) {
			leftBit = R & mask;
			R <<= 1;
			if(leftBit != 0)
				R += 1;
		}
		R ^= L;
		L = V;
	}
	unsigned long long res = (unsigned long long) L << 32;
	res += R;
	result[index] = res;
}

__global__ void gostDeciphered(unsigned long long *data, unsigned long long *result) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int L = ((data[index] & 0xFFFFFFFF00000000) >> 32);
	unsigned int R = (data[index] & 0x00000000FFFFFFFF);
	const long long number = 4294967296L;
	int j = 0;
	for(int i = 1; i <= 32; i ++) {
	    unsigned int V = L;
	    if(i <= 8)
			j = (i - 1) % 8;
		else
			j = (32 - i) % 8;
		long long buf = L + gKey[j];
		while(buf >= number)
			buf = buf - number;
		L = buf;
		F(&L);
		unsigned long long mask = 0x80000000;
		unsigned int leftBit;
		for(int k = 0; k < 11; k ++) {
			leftBit = L & mask;
			L <<= 1;
			if(leftBit != 0)
				L += 1;
		}
		L ^= R;
		R = V;
	}
	unsigned long long res = (unsigned long long) L << 32;
	res += R;
	result[index] = res;
}

void createTable()
{
	short int table[8][16];
	int a = 5; 
	int c = 7;
	int m = 16;
	for(int i = 0; i < 8; i ++) {
		int t = i;
		table[i][0] = t;
		for(int j = 1; j < 16; j ++) {
			t = (a * t + c) % m;
			table[i][j] = t;
		}
	}
	for(int i = 0; i < 8; i ++ ) {
		for(int j = 0; j < 16; j ++)
			cout<<table[i][j] <<"\t";
		cout<<endl;
	}
	cudaMemcpyToSymbol(gTable[0], table[0], sizeof(short int) * 16 * 8, 0, cudaMemcpyHostToDevice);
}

 extern "C" void launch_gost(string fName, string kName, bool mode) 
 {
	 cout<<"Run gost algorithm\n";
	 cout<<"File name: "<<fName<<endl;
	 cout<<"Key name: "<<kName<<endl;
	 unsigned int key[8];
	 //read key
	 FILE *iKey;
	 iKey = fopen(kName.c_str(), "rb");
	 fread(&key[0], sizeof(key[0]), 8, iKey);
	 fclose(iKey);
	 //write key 
	 cout<<"Key: ";
	 for(int i = 0; i < 8; i ++)
		 cout<<key[i]<<" ";
	 cout<<endl;
	 //read data and run crypt algorithm
	 FILE *iFile;
	 FILE *oFile;
	 unsigned long long data[SIZE];
	 unsigned long long result[SIZE];
	 unsigned long long *gResult;
	 unsigned long long *gData;
	 cudaMemcpyToSymbol(gKey, key , sizeof(unsigned int) * 8, 0, cudaMemcpyHostToDevice);
	 createTable();
	 iFile = fopen(fName.c_str(),"rb");
	 oFile = fopen((fName + "o").c_str(),"wb");
	 fseek(iFile, 0, SEEK_END);
	 long long fSize = ftell(iFile);
	 fseek(iFile, 0, SEEK_SET);
	 cout<<"File size = "<<fSize<<endl;
	 cudaMalloc((void **) &gResult, sizeof(unsigned long long) * SIZE);
	 cudaMalloc((void **) &gData, sizeof(unsigned long long) * SIZE);
	 while(fSize >= SIZE) {
	 fread(&data[0], sizeof(data[0]), SIZE,iFile);
	 fSize -= (SIZE * sizeof(unsigned long long));
     cudaMemcpy(gData, data, sizeof(unsigned long long) * SIZE, cudaMemcpyHostToDevice);
	 if(mode)
	   gostEncrypt<<<dim3(10, 1, 1),dim3(128, 1, 1)>>>(gData, gResult);
	 else
	   gostDeciphered<<<dim3(10, 1, 1),dim3(128, 1, 1)>>>(gData, gResult);
	 cudaEvent_t syncEvent;
     cudaEventCreate(&syncEvent);    //Создаем event
     cudaEventRecord(syncEvent, 0);  //Записываем event
     cudaEventSynchronize(syncEvent);  //Синхронизируем event
     cudaMemcpy((void *) &result, gResult, SIZE * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	 fwrite(&result[0], sizeof(result[0]), SIZE, oFile);
     }
	 cudaFree(gResult);
	 cudaFree(gData);
	 int count = 0;
     while(fread(&data[count], sizeof(data[count]), 1,iFile))
		 count++;
	 cudaMalloc((void **) &gResult, sizeof(unsigned long long) * count);
	 cudaMalloc((void **) &gData, sizeof(unsigned long long) * count);
	 cudaMemcpy(gData, data, sizeof(unsigned long long) * count, cudaMemcpyHostToDevice);
	 if(mode)
	   gostEncrypt<<<dim3(10, 1, 1),dim3(count/10, 1, 1)>>>(gData, gResult);
	 else
	   gostDeciphered<<<dim3(10, 1, 1),dim3(count/10, 1, 1)>>>(gData, gResult);
	 cudaEvent_t syncEvent;
     cudaEventCreate(&syncEvent);    //Создаем event
     cudaEventRecord(syncEvent, 0);  //Записываем event
     cudaEventSynchronize(syncEvent);  //Синхронизируем event
     cudaMemcpy((void *) &result, gResult, count * sizeof(unsigned long long),cudaMemcpyDeviceToHost);
	 fwrite(&result[0], sizeof(result[0]), count, oFile);
	 cudaFree(gResult);
	 cudaFree(gData);
	 fclose(iFile);
	 fclose(oFile);
	 cout<<"Finish\n";
 }