#include "runthread.h"

RunThread::RunThread(string fName, string kName, bool m, int alg):
    fileName(fName),
	keyName(kName),
	mode(m),
	algorithm(alg)
{

}

RunThread::~RunThread()
{
	cout<<"Thread Destroy\n";
}

void RunThread::run()
{
	cout<<"Thread Start\n";
	switch(algorithm) {
	case 1: launch_gost(fileName, keyName, mode);
		break;
	case 2: launch_aes(fileName, keyName, mode, 256);
		break;
	}
	cout<<"Thread Finish\n";
}
