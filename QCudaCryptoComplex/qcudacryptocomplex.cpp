#include "qcudacryptocomplex.h"
#include <qfiledialog.h>
#include "runthread.h"
#include <iostream>

QCudaCryptoComplex::QCudaCryptoComplex(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
}

QCudaCryptoComplex::~QCudaCryptoComplex()
{

}

void QCudaCryptoComplex::on_pushButton_2_clicked()
{
	fileName = QFileDialog::getOpenFileName(this, tr("Âûבונטעו פאיכ"), "C:/Users/164981/Desktop", tr("All Files (*.*)"));
	std::cout<<"Open\n";
	this->ui.lineEdit->setText(fileName);
	std::cout<<"Print\n";
}

void QCudaCryptoComplex::on_pushButton_3_clicked()
{
	keyFileName = QFileDialog::getOpenFileName(this, tr("Âûבונטעו ךכ‏ק"), "C:/Users/164981/Desktop", tr("All Files (*.txt)"));
	std::cout<<"open\n";
	ui.lineEdit_2->setText(keyFileName);
	std::cout<<"print\n";
}

void QCudaCryptoComplex::on_pushButton_clicked()
{

}

void QCudaCryptoComplex::on_pushButton_4_clicked()
{
	bool type = false;
	if(ui.radioButton_7->isChecked())
		type = true;
	RunThread *thread= new RunThread(ui.lineEdit->text().toStdString(), ui.lineEdit_2->text().toStdString(), type);
	thread->start();
	QObject::connect(thread, SIGNAL(finished()), thread, SLOT(deleteLater()));
}
