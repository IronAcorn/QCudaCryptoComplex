#include "qcudacryptocomplex.h"
#include <qfiledialog.h>
#include "runthread.h"
#include <iostream>

QCudaCryptoComplex::QCudaCryptoComplex(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	ui.radioButton_2->click();
	ui.radioButton_7->click();
}

QCudaCryptoComplex::~QCudaCryptoComplex()
{

}

void QCudaCryptoComplex::on_pushButton_clicked()
{

}

void QCudaCryptoComplex::on_pushButton_2_clicked()
{
	QString fileName = QFileDialog::getOpenFileName(this, tr("Âûבונטעו פאיכ"), "C:/Users/164981/Desktop", tr("All Files (*.*)"));
	std::cout<<"Open\n";
	this->ui.lineEdit->setText(fileName);
	std::cout<<"Print\n";
}

void QCudaCryptoComplex::on_pushButton_3_clicked()
{
	QString keyFileName = QFileDialog::getOpenFileName(this, tr("Âûבונטעו ךכ‏ק"), "C:/Users/164981/Desktop", tr("All Files (*.txt)"));
	std::cout<<"open\n";
	ui.lineEdit_2->setText(keyFileName);
	std::cout<<"print\n";
}

void QCudaCryptoComplex::on_pushButton_4_clicked()
{
	if(ui.radioButton_2->isChecked()) {
	    RunThread *thread= new RunThread(ui.lineEdit->text().toStdString(), ui.lineEdit_2->text().toStdString(), mode, 1);
	    thread->start();
	    QObject::connect(thread, SIGNAL(finished()), thread, SLOT(deleteLater()));
	}
	if(ui.radioButton->isChecked()) {
	    RunThread *thread= new RunThread(ui.lineEdit->text().toStdString(), ui.lineEdit_2->text().toStdString(), mode, 2);
	    thread->start();
	    QObject::connect(thread, SIGNAL(finished()), thread, SLOT(deleteLater()));
	}

}

void QCudaCryptoComplex::on_radioButton_6_clicked()
{
	ui.radioButton_7->setChecked(false);
	mode = false;
}

void QCudaCryptoComplex::on_radioButton_7_clicked()
{
	ui.radioButton_6->setChecked(false);
	mode = true;
}
