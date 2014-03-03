#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <string>
using namespace std;

#define SIZE 13107200

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

__global__ void gostEncrypt(unsigned long long *data) 
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
	data[index] = res;
}

__global__ void gostDeciphered(unsigned long long *data) 
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
	data[index] = res;
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
	 unsigned int key[8];
	 FILE *iKey;
	 FILE *iFile;
	 FILE *oFile;
	 unsigned long long *buf = new unsigned long long [SIZE];
	 unsigned long long *data = new unsigned long long [SIZE];
	 //unsigned long long *gResult;
	 unsigned long long *gData;
	 int count = 0;
	 cout<<"Run gost algorithm\n";
	 cout<<"File name: "<<fName<<endl;
	 cout<<"Key name: "<<kName<<endl;
	 cout<<"Mode: "<<mode<<endl;
	 //read key
	 iKey = fopen(kName.c_str(), "rb");
	 fread(&key[0], sizeof(key[0]), 8, iKey);
	 fclose(iKey);
	 //write key 
	 cout<<"Key: ";
	 for(int i = 0; i < 8; i ++)
		 cout<<key[i]<<" ";
	 cout<<endl;
	 //read data and run crypt algorithm
	 cudaMemcpyToSymbol(gKey, key , sizeof(unsigned int) * 8, 0, cudaMemcpyHostToDevice);
	 createTable();
	 iFile = fopen(fName.c_str(),"rb");
	 int pos = fName.find_last_of("/\\");
	 if(mode)
	     fName.insert(pos + 1, "encrypt");
	 else
		 fName.insert(pos + 1, "decrypt");
	 oFile = fopen((fName).c_str(),"wb");
	 //cudaMalloc((void **) &gResult, sizeof(unsigned long long) * SIZE);
	 cudaMalloc((void **) &gData, sizeof(unsigned long long) * SIZE);
	 while(count = fread(&buf[0], sizeof(buf[0]), SIZE, iFile)) {
         cudaMemcpy(gData, &buf[0], sizeof(unsigned long long) * count, cudaMemcpyHostToDevice);
	     if(mode)
	         gostEncrypt<<<dim3(1000, 1, 1),dim3(512, 1, 1)>>>(gData);
	     else
	         gostDeciphered<<<dim3(1000, 1, 1),dim3(512, 1, 1)>>>(gData);
		 //sync
	     cudaEvent_t syncEvent;
         cudaEventCreate(&syncEvent);    
         cudaEventRecord(syncEvent, 0);  
         cudaEventSynchronize(syncEvent);  
         cudaMemcpy(&data[0], gData, count * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	     fwrite(&data[0], sizeof(data[0]), count, oFile);
     }
	 //cudaFree(gResult);
	 cudaFree(gData);
	 delete[] buf;
	 delete[] data;
	 fclose(iFile);
	 fclose(oFile);
	 cout<<"Finish\n";
 } 

 /*-----------------------------------------------------------AES-----------------------------------------------------------------------*/

 __constant__ unsigned char gSubTable[256];
 __constant__ unsigned char gAesKey[32];
 __constant__ unsigned int gWords[60];
 
 struct node {
	 unsigned char dta[4][4];
 };

__device__ void subBytes(node *data) 
{
	for(int i = 0; i < 4; i ++)
		for(int j = 0; j < 4; j ++)
			data->dta[j][i] = gSubTable[data->dta[j][i]];
}

__device__ void shiftRows(node *data) 
{
	unsigned char buf;
	for(int i = 0; i < 4; i ++) {
			for(int k = 0; k < i; k ++) {
				buf = data->dta[0][i];
				data->dta[0][i] = data->dta[1][i];
				data->dta[1][i] = data->dta[2][i];
				data->dta[2][i] = data->dta[3][i];
				data->dta[3][i] = buf;
			}
	}
}

__device__ unsigned char multiply(unsigned char p, unsigned char v)
{
	unsigned char mask = 0x80;
	if(p == 1)
		return v;
	if(p == 2) {
		if((v & mask) != 0) {
			v <<= 1;
			v ^= 0x1B;
			return v;
		}
		else
			return v <<= 1;
	}
	if(p == 3) {
		unsigned char buf = v;
		if((v & mask) != 0) {
			v <<= 1;
			v ^= 0x1B;
			v ^= buf;
			return v;
		}
		else {
			v <<= 1;
			v ^= buf;
			return v;
		}
			

	}
}

__device__ void mixColumns(node *data) 
{
	unsigned char b[4][4] = { {2, 3, 1, 1},
	                          {1, 2, 3, 1},
	                          {1, 1, 2, 3},
	                          {3, 1, 1, 2}};
	unsigned char r[4][4] = {0};
	for(int k = 0; k < 4; k ++) {
	    for(int i = 0; i < 4; i ++) {
		    for(int j = 0; j < 4; j ++) {
			    r[i][k] ^= multiply(b[i][j], data->dta[k][j]);
		    }
	    }
	}

	for(int i = 0; i < 4; i ++)
		for(int j = 0; j < 4; j ++)
			data->dta[j][i] = r[i][j];

}

__device__ void addRoundKey(node *data, int r)
{
	for(int j = 0; j < 4; j ++) {
		unsigned int buf = 0;
		for(int i = 3; i >= 0; i --) {
			buf += (data->dta[j][i] << 8 * i);
		}
		buf ^= gWords[4*r + j];
		for(int i = 0; i < 4; i ++) {
			data->dta[j][i] = buf & 0xFF;
			buf >>= 8;
		}
	}
			
}


 __global__ void aesEncrypt(node *data, int rounds)
 {
	 int index = blockDim.x * blockIdx.x + threadIdx.x;
	 addRoundKey(&data[index], 0);
	 for(int i = 1; i <= rounds; i ++) {
	     subBytes(&data[index]);
	     shiftRows(&data[index]);
		 if(i != rounds)
	         mixColumns(&data[index]);
		 addRoundKey(&data[index], i);
	 }
	 //result[index] = data[index];
 }

__device__ void invShiftRows(node * data)
{
	unsigned char buf;
	for(int i = 0; i < 4; i ++) {
			for(int k = 0; k < i; k ++) {
				buf = data->dta[3][i];
				data->dta[3][i] = data->dta[2][i];
				data->dta[2][i] = data->dta[1][i];
				data->dta[1][i] = data->dta[0][i];
				data->dta[0][i] = buf;
			}
	}
}

__device__ unsigned char invMultiply(unsigned char b, unsigned char a)
{
   unsigned char p = 0;
   unsigned char counter;
   unsigned char hi_bit_set;
   for (counter = 0; counter < 8; counter++) {
      if ((b & 1) != 0) {
         p ^= a;
      }
      hi_bit_set = (unsigned char) (a & 0x80);
      a <<= 1;
      if (hi_bit_set != 0) {
         a ^= 0x1b; /* x^8 + x^4 + x^3 + x + 1 */
      }
      b >>= 1;
   }
   return p;
}

__device__ void invMixColumns(node *data)
{
	unsigned char b[4][4] = { {0x0e, 0x0b, 0x0d, 0x09},
	                          {0x09, 0x0e, 0x0b, 0x0d},
	                          {0x0d, 0x09, 0x0e, 0x0b},
	                          {0x0b, 0x0d, 0x09, 0x0e}};
	unsigned char r[4][4] = {0};
	for(int k = 0; k < 4; k ++) {
	    for(int i = 0; i < 4; i ++) {
		    for(int j = 0; j < 4; j ++) {
			    r[i][k] ^= invMultiply(b[i][j], data->dta[k][j]);
		    }
	    }
	}

	for(int i = 0; i < 4; i ++)
		for(int j = 0; j < 4; j ++)
			data->dta[j][i] = r[i][j];
}

 __global__ void aesDeciphered(node *data, int rounds)
 {
	 int index = blockDim.x * blockIdx.x + threadIdx.x;
	 addRoundKey(&data[index], rounds);
	 for(int i = rounds - 1; i >= 0; i --) {
		 invShiftRows(&data[index]);
		 subBytes(&data[index]);
		 addRoundKey(&data[index], i);
		 if(i != 0)
			 invMixColumns(&data[index]);
		 
	 }
	 //result[index] = data[index];

 }

 unsigned int rc(int n) 
 {
	 unsigned char rCon[256] = {
     0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 
     0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 
     0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 
     0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 
     0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 
     0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 
     0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 
     0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 
     0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 
     0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 
     0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 
     0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 
     0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 
     0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 
     0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 
     0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d};
	 return rCon[n];
 }

 unsigned int subWord(unsigned char *table, unsigned int n) 
 {
	unsigned int res = 0;
	unsigned int mask = 0xFF000000;
	for(int i = 3; i >=0; i --) {
		res <<= 8;
		unsigned short buf = table[(n & mask) >> 8 * i];
		mask >>= 8;
		res += buf;
	}
	return res;
 }

 unsigned int rotWord(unsigned int n)
 {
	 unsigned int buf = n & 0xFF;
	 n >>= 8;
	 buf <<= 24;
	 n += buf;
	 return n;
 }

 int rounds(int keySize)
 {
    switch(keySize) {
	    case 128: 
			return 10;
		case 192:
			return 12;
		case 256:
			return 14;
		default:
			return -1;
	}
 }

 void createWordsAndTable(unsigned char *key, int keySize, bool mode) 
 {
	 unsigned char table[] = {
		0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76, 
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0, 
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15, 
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75, 
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84, 
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf, 
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8, 
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2, 
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73, 
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb, 
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, 
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08, 
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a, 
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e, 
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf, 
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
    };
	unsigned char table2[] = {
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
    if(mode)
	    cudaMemcpyToSymbol(gSubTable, table, sizeof(table), 0, cudaMemcpyHostToDevice);
	else
		cudaMemcpyToSymbol(gSubTable, table2, sizeof(table2), 0, cudaMemcpyHostToDevice);
    unsigned int r = rounds(keySize);
	int nK = keySize / 32;
	unsigned int *words = new unsigned int[4 * (r + 1)];
	for(int i = 0; i < 4 * (r + 1); i ++)
		words[i] = 0;
	for(int i = 0; i < nK; i ++) {
		for(int j = 0; j < 4; j ++) 
			words[i] += ((key[4 * i + j]) << (8 * j));
	}
	for(int i = nK; i < 4 * (r + 1); i ++) {
		unsigned int t = words[i - 1];
		if(i % nK == 0) 
			t = subWord(table, rotWord(t)) ^ rc(i/nK);
		if((nK == 8) && (i % nK == 4))
			t = subWord(table, t);
		words[i] = (unsigned int) words[i - nK] ^ t;
	 }
	 cudaMemcpyToSymbol(gWords, words, sizeof(unsigned int) * (4 * (r + 1)), 0, cudaMemcpyHostToDevice);
 }
 


 extern "C" void launch_aes(string fName, string kName, bool mode, int keySize) 
 {
	 FILE *iKey;
	 FILE *iFile;
	 FILE *oFile;
	 unsigned char key[32];
	 int r = rounds(keySize);
	 struct::node *buf = new node[SIZE];
	 struct::node *data = new node[SIZE];
	 struct::node *gData;
	 //struct::node *gResult;
	 int count = 0;
	 cout<<fName<<" "<<kName<<" "<<mode<<" "<<keySize<<endl;
	 //read key
	 iKey = fopen(kName.c_str(), "rb");
	 fread(key, sizeof(unsigned char), keySize / 4, iKey);
	 fclose(iKey);
	 cudaMemcpyToSymbol(gAesKey, key, keySize / 4, 0,cudaMemcpyHostToDevice);
	 iFile = fopen(fName.c_str(), "rb");
	 int pos = fName.find_last_of("/\\");
	 if(mode) {
	     fName.insert(pos + 1, "encrypt");
		 createWordsAndTable(key, keySize, true);
	 }
	 else {
		 fName.insert(pos + 1, "decrypt");
		 createWordsAndTable(key, keySize, false);
	 }
	 oFile = fopen(fName.c_str(), "wb");
	 cudaMalloc((void **) &gData, sizeof(node) * SIZE);
	 //cudaMalloc((void **) &gResult, sizeof(node) *SIZE);
	 while(count = fread(buf, sizeof(buf[0]), SIZE, iFile)) {
		 cudaMemcpy(gData, &buf[0], sizeof(buf[0]) * count, cudaMemcpyHostToDevice);
		 if(mode)
			 aesEncrypt<<<dim3(1000, 1, 1), dim3(512, 1, 1)>>>(gData, r);
		 else
			 aesDeciphered<<<dim3(1000, 1, 1), dim3(512, 1, 1)>>>(gData, r);
		 //synchronize
		 cudaEvent_t syncEvent;
		 cudaEventCreate(&syncEvent);
		 cudaEventRecord(syncEvent, 0);
		 cudaEventSynchronize(syncEvent);
		 cudaMemcpy(data, gData, sizeof(data[0]) * count, cudaMemcpyDeviceToHost);
		 fwrite(data, sizeof(data[0]), count, oFile);
	 }
	 cudaFree(gData);
	 //cudaFree(gResult);
	 delete[] buf;
	 delete[] data;
	 fclose(iFile);
	 fclose(oFile);
	 cout<<"Finish\n";
 }
