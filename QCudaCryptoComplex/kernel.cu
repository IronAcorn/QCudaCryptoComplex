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

__global__ void gostEncrypt(unsigned long long *data, unsigned long long *result) 
{

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

__global__ void gostDeciphered(unsigned long long *data, unsigned long long *result) 
{
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
	 int pos = fName.find_last_of("/\\");
	 if(mode)
	     fName.insert(pos + 1, "encrypt");
	 else
		 fName.insert(pos + 1, "decrypt");
	 oFile = fopen((fName).c_str(),"wb");
	 cudaMalloc((void **) &gResult, sizeof(unsigned long long) * SIZE);
	 cudaMalloc((void **) &gData, sizeof(unsigned long long) * SIZE);
	 int count = 0;
	 while(count = fread(&data[0], sizeof(data[0]), SIZE, iFile)) {
     cudaMemcpy(gData, data, sizeof(unsigned long long) * count, cudaMemcpyHostToDevice);
	 if(mode)
	     gostEncrypt<<<dim3(10, 1, 1),dim3(count / 10, 1, 1)>>>(gData, gResult);
	 else
	     gostDeciphered<<<dim3(10, 1, 1),dim3(count / 10, 1, 1)>>>(gData, gResult);
	 cudaEvent_t syncEvent;
     cudaEventCreate(&syncEvent);    //Создаем event
     cudaEventRecord(syncEvent, 0);  //Записываем event
     cudaEventSynchronize(syncEvent);  //Синхронизируем event
     cudaMemcpy((void *) &result, gResult, count * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	 fwrite(&result[0], sizeof(result[0]), count, oFile);
     }
	 cudaFree(gResult);
	 cudaFree(gData);
	 fclose(iFile);
	 fclose(oFile);
	 cout<<"Finish\n";
 }

 /*-----------------------------------------------------------AES-----------------------------------------------------------------------*/

 __constant__ unsigned char gSubTable[256];
 __constant__ unsigned char gAesKey[32];
  struct node {
	 unsigned char dta[4][4];
 };

__device__ void subBytes(node *data) 
{
	for(int i = 0; i < 4; i ++)
		for(int j = 0; j < 4; j ++)
			data->dta[i][j] = gSubTable[data->dta[i][j]];
}

__device__ void shiftRows(node *data) 
{
	unsigned char buf;
	for(int i = 0; i < 4; i ++)
		for(int j = 0; j < 4; j ++) {
			for(int k = 0; k < i; k ++) {
				buf = data->dta[i][0];
				data->dta[i][0] = data->dta[i][1];
				data->dta[i][1] = data->dta[i][2];
				data->dta[i][2] = data->dta[i][3];
				data->dta[i][3] = buf;
			}
		}

}

__device__ void mixColumns(node *data) 
{
	long long mod;
	for(int j = 0; j < 4; j ++)
		for(int i = 0; i < 4; i ++) {
			mod = (long long) powf(4, data->dta[i][j]) + 1;
			int buf = data->dta[i][j];
			data->dta[i][j] = (unsigned char) ((long long) 3 * powf(3, buf) + powf(2, buf) + buf + 2) % mod;
		}
}


 __global__ void aesEncrypt(node *data, node *result)
 {
	 int index = blockDim.x * blockIdx.x + threadIdx.x;
	 subBytes(&data[index]);
	 shiftRows(&data[index]);
	 mixColumns(&data[index]);
 }

 void createSubTable() {
	 unsigned char table[] = {
		 0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
        0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
        0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
        0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
        0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
        0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
        0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
        0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
        0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
        0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
        0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
        0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
        0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
        0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
        0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
        0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
    };
	cudaMemcpyToSymbol(gSubTable, table, sizeof(table), cudaMemcpyHostToDevice);
 }
 


 extern "C" void launch_aes(string fName, string kName, bool mode) 
 {
	 FILE *iKey;
	 unsigned char key[32];
	 //read key
	 iKey = fopen(kName.c_str(), "rb");
	 fread(key, sizeof(unsigned char), 32, iKey);
	 fclose(iKey);
	 cudaMemcpyToSymbol(gAesKey, key, sizeof(key), cudaMemcpyHostToDevice);
	 FILE *iFile, *oFile;
	 struct::node data[SIZE];
	 struct::node result[SIZE];
	 struct::node *gData;
	 struct::node *gResult;
	 iFile = fopen(fName.c_str(), "rb");
	 int pos = fName.find_last_of("/\\");
	 if(mode) {
	     fName.insert(pos + 1, "encrypt");
		 createSubTable();
	 }
	 else
		 fName.insert(pos + 1, "decrypt");
	 oFile = fopen(fName.c_str(), "wb");
	 int count = 0;
	 cudaMalloc((void **) &gData, sizeof(node) * SIZE);
	 cudaMalloc((void **) &gResult, sizeof(node) *SIZE);
	 while(count = fread(data, sizeof(data[0]), SIZE, iFile)) {
		 cudaMemcpy(gData, data, sizeof(data[0]) * count, cudaMemcpyHostToDevice);
		 if(mode)
			 aesEncrypt<<<dim3(10, 1 ,1), dim3(count / 10, 1, 1)>>>(gData, gResult);
		 //synchronize
		 cudaEvent_t syncEvent;
		 cudaEventCreate(&syncEvent);
		 cudaEventRecord(syncEvent, 0);
		 cudaEventSynchronize(syncEvent);
		 cudaMemcpy(result, gResult, sizeof(result[0]) * count, cudaMemcpyDeviceToHost);
		 fwrite(result, sizeof(result[0]), count, oFile);
	 }
	 cudaFree(gData);
	 cudaFree(gResult);
	 fclose(iFile);
	 fclose(oFile);
	 cout<<"Finish\n";
 }

