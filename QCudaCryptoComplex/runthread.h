#ifndef RUNTHREAD_H
#define RUNTHREAD_H

#include <QThread>
#include <string>
#include <iostream>
using namespace std;

extern "C" void launch_gost(string, string, bool);

class RunThread : public QThread
{
	Q_OBJECT

public:
	RunThread(string, string, bool);
	~RunThread();
	void run();

private:
	string fileName;
	string keyName;
	bool mode;
	
};

#endif // RUNTHREAD_H