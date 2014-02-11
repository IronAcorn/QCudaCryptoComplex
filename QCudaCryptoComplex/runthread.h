#ifndef RUNTHREAD_H
#define RUNTHREAD_H

#include <QThread>
#include <string>
#include <iostream>
using namespace std;

extern "C" void launch_gost(string, string, bool);
extern "C" void launch_aes(string, string, bool, int);

class RunThread : public QThread
{
	Q_OBJECT

public:
	RunThread(string, string, bool, int);
	~RunThread();
	void run();

private:
	string fileName;
	string keyName;
	bool mode;
	int algorithm;
	
};

#endif // RUNTHREAD_H
