#include "qcudacryptocomplex.h"
#include <qfiledialog.h>
#include "runthread.h"
#include <iostream>

QCudaCryptoComplex::QCudaCryptoComplex(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	ui.radioButton_8->hide();
	ui.radioButton_9->hide();
	ui.radioButton_10->hide();
	ui.radioButton_2->click();
	ui.radioButton_7->click();
	group = new QButtonGroup(this);
	group->addButton(ui.radioButton_8);
	group->addButton(ui.radioButton_9);
	group->addButton(ui.radioButton_10);
    group2 = new QButtonGroup(this);
	group2->addButton(ui.radioButton);
	group2->addButton(ui.radioButton_2);
	group2->addButton(ui.radioButton_3);
	group2->addButton(ui.radioButton_4);
	group2->addButton(ui.radioButton_5);
	group3 = new QButtonGroup(this);
	group3->addButton(ui.radioButton_6);
	group3->addButton(ui.radioButton_7);
}

QCudaCryptoComplex::~QCudaCryptoComplex()
{
	delete group;
	delete group2;
	delete group3;
}

void QCudaCryptoComplex::on_pushButton_clicked()
{

}

void QCudaCryptoComplex::on_pushButton_2_clicked()
{
	QString fileName = QFileDialog::getOpenFileName(this, tr("Âûבונטעו פאיכ"), "D:/", tr("All Files (*.*)"));
	ui.lineEdit->setText(fileName);
}

void QCudaCryptoComplex::on_pushButton_3_clicked()
{
	QString keyFileName = QFileDialog::getOpenFileName(this, tr("Âûבונטעו ךכ‏ק"), "D:/", tr("All Files (*.txt)"));
	ui.lineEdit_2->setText(keyFileName);
}

void QCudaCryptoComplex::on_pushButton_4_clicked()
{
	cout<<"mode: "<<mode<<" alg: "<<alg<<" size: "<<keySize<<endl;
	timer = new QTimer(this);
	QObject::connect(timer, SIGNAL(timeout()), this, SLOT(timeout()));
	time = 0;
	timer->start(200);
	RunThread *thread= new RunThread(ui.lineEdit->text().toStdString(), ui.lineEdit_2->text().toStdString(), mode, alg, keySize);
	thread->start();
	QObject::connect(thread, SIGNAL(finished()), thread, SLOT(deleteLater()));
    QObject::connect(thread, SIGNAL(finished()), this, SLOT(finish()));
}

void QCudaCryptoComplex::on_radioButton_toggled(bool value)
{
	if(value) {
		alg = 2;
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

void QCudaCryptoComplex::on_radioButton_2_toggled(bool value)
{
	if(value)
		alg = 1;
}

void QCudaCryptoComplex::on_radioButton_7_toggled(bool value)
{
	if(value)
		mode = true;
	else
		mode = false;
	cout<<mode<<endl;
}

void QCudaCryptoComplex::on_radioButton_8_toggled(bool value)
{
	if(value)
		keySize = 256;
}

void QCudaCryptoComplex::on_radioButton_9_toggled(bool value)
{
	if(value)
		keySize = 192;
}

void QCudaCryptoComplex::on_radioButton_10_toggled(bool value)
{
	if(value)
		keySize = 128;
}

void QCudaCryptoComplex::timeout()
{
	time += 200;
	ui.lcdNumber->display(time);
}

void QCudaCryptoComplex::finish()
{
	timer->stop();
	delete timer;
	cout<<"Time: "<<time<<endl;
}
