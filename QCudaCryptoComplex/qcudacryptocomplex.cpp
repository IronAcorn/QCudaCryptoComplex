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
	QObject::connect(ui.radioButton, SIGNAL(toggled(bool)), this, SLOT(showType(bool)));
	ui.radioButton_8->hide();
	ui.radioButton_9->hide();
	ui.radioButton_10->hide();

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
	int size = 0;
		if(ui.radioButton_10->isChecked())
			size = 128;
		if(ui.radioButton_9->isChecked())
			size = 192;
		if(ui.radioButton_8->isChecked())
			size = 256;
	if(ui.radioButton_2->isChecked()) {
	    RunThread *thread= new RunThread(ui.lineEdit->text().toStdString(), ui.lineEdit_2->text().toStdString(), mode, 1, size);
	    thread->start();
	    QObject::connect(thread, SIGNAL(finished()), thread, SLOT(deleteLater()));
	}
	if(ui.radioButton->isChecked()) {
	    RunThread *thread= new RunThread(ui.lineEdit->text().toStdString(), ui.lineEdit_2->text().toStdString(), mode, 2, size);
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

void QCudaCryptoComplex::showType(bool mode)
	
{
	cout<<mode<<endl;
	if(mode) {
	    ui.radioButton_8->show();
	    ui.radioButton_9->show();
	    ui.radioButton_10->show();
	    ui.radioButton_10->click();
	} else {
		ui.radioButton_8->hide();
	    ui.radioButton_9->hide();
	    ui.radioButton_10->hide();
	}
}
