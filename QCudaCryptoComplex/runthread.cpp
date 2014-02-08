#include "runthread.h"

RunThread::RunThread(string fName, string kName, bool m):
    fileName(fName),
	keyName(kName),
	mode(m)
{

}

RunThread::~RunThread()
{
	cout<<"Thread Destroy\n";
}

void RunThread::run()
{
	cout<<"Thread Start\n";
	launch_gost(fileName, keyName, mode);
	cout<<"Thread Finish\n";
}
