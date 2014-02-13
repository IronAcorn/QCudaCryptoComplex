#include "runthread.h"

RunThread::RunThread(string fName, string kName, bool m, int alg, int l):
    fileName(fName),
	keyName(kName),
	mode(m),
	algorithm(alg),
	length(l)
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
	case 2: launch_aes(fileName, keyName, mode, length);
		break;
	}
	cout<<"Thread Finish\n";
}
