#ifndef QCUDACRYPTOCOMPLEX_H
#define QCUDACRYPTOCOMPLEX_H

#include <QtWidgets/QMainWindow>
#include "ui_qcudacryptocomplex.h"
#include <QTimer>
#include <QButtonGroup>
#include <qstring.h>
#include <qevent.h>
#include <string>
using namespace std;

class QCudaCryptoComplex : public QMainWindow
{
	Q_OBJECT

public:
	QCudaCryptoComplex(QWidget *parent = 0);
	~QCudaCryptoComplex();
public slots:
	void on_pushButton_clicked();
	void on_pushButton_2_clicked();
	void on_pushButton_3_clicked();
	void on_pushButton_4_clicked();
	void on_radioButton_toggled(bool);
	void on_radioButton_2_toggled(bool);
	void on_radioButton_7_toggled(bool);
	void on_radioButton_8_toggled(bool);
	void on_radioButton_9_toggled(bool);
	void on_radioButton_10_toggled(bool);
	void timeout();
	void finish();
private:
	Ui::QCudaCryptoComplexClass ui;
	QTimer *timer;
	QButtonGroup *group;
	QButtonGroup *group2;
	QButtonGroup *group3;
	int time;
	bool mode;
	int alg;
	int keySize;
};

#endif // QCUDACRYPTOCOMPLEX_H
