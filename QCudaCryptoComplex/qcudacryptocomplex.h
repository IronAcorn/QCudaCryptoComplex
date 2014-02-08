#ifndef QCUDACRYPTOCOMPLEX_H
#define QCUDACRYPTOCOMPLEX_H

#include <QtWidgets/QMainWindow>
#include "ui_qcudacryptocomplex.h"
#include <qstring.h>
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
	void on_radioButton_6_clicked();
	void on_radioButton_7_clicked();

private:
	Ui::QCudaCryptoComplexClass ui;
	bool mode;
};

#endif // QCUDACRYPTOCOMPLEX_H
