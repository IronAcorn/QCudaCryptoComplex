/****************************************************************************
** Meta object code from reading C++ file 'qcudacryptocomplex.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.1.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../qcudacryptocomplex.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'qcudacryptocomplex.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.1.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_QCudaCryptoComplex_t {
    QByteArrayData data[14];
    char stringdata[279];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    offsetof(qt_meta_stringdata_QCudaCryptoComplex_t, stringdata) + ofs \
        - idx * sizeof(QByteArrayData) \
    )
static const qt_meta_stringdata_QCudaCryptoComplex_t qt_meta_stringdata_QCudaCryptoComplex = {
    {
QT_MOC_LITERAL(0, 0, 18),
QT_MOC_LITERAL(1, 19, 21),
QT_MOC_LITERAL(2, 41, 0),
QT_MOC_LITERAL(3, 42, 23),
QT_MOC_LITERAL(4, 66, 23),
QT_MOC_LITERAL(5, 90, 23),
QT_MOC_LITERAL(6, 114, 22),
QT_MOC_LITERAL(7, 137, 24),
QT_MOC_LITERAL(8, 162, 24),
QT_MOC_LITERAL(9, 187, 24),
QT_MOC_LITERAL(10, 212, 24),
QT_MOC_LITERAL(11, 237, 25),
QT_MOC_LITERAL(12, 263, 7),
QT_MOC_LITERAL(13, 271, 6)
    },
    "QCudaCryptoComplex\0on_pushButton_clicked\0"
    "\0on_pushButton_2_clicked\0"
    "on_pushButton_3_clicked\0on_pushButton_4_clicked\0"
    "on_radioButton_toggled\0on_radioButton_2_toggled\0"
    "on_radioButton_7_toggled\0"
    "on_radioButton_8_toggled\0"
    "on_radioButton_9_toggled\0"
    "on_radioButton_10_toggled\0timeout\0"
    "finish\0"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_QCudaCryptoComplex[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      12,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   74,    2, 0x0a,
       3,    0,   75,    2, 0x0a,
       4,    0,   76,    2, 0x0a,
       5,    0,   77,    2, 0x0a,
       6,    1,   78,    2, 0x0a,
       7,    1,   81,    2, 0x0a,
       8,    1,   84,    2, 0x0a,
       9,    1,   87,    2, 0x0a,
      10,    1,   90,    2, 0x0a,
      11,    1,   93,    2, 0x0a,
      12,    0,   96,    2, 0x0a,
      13,    0,   97,    2, 0x0a,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,    2,
    QMetaType::Void, QMetaType::Bool,    2,
    QMetaType::Void, QMetaType::Bool,    2,
    QMetaType::Void, QMetaType::Bool,    2,
    QMetaType::Void, QMetaType::Bool,    2,
    QMetaType::Void, QMetaType::Bool,    2,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void QCudaCryptoComplex::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        QCudaCryptoComplex *_t = static_cast<QCudaCryptoComplex *>(_o);
        switch (_id) {
        case 0: _t->on_pushButton_clicked(); break;
        case 1: _t->on_pushButton_2_clicked(); break;
        case 2: _t->on_pushButton_3_clicked(); break;
        case 3: _t->on_pushButton_4_clicked(); break;
        case 4: _t->on_radioButton_toggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 5: _t->on_radioButton_2_toggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 6: _t->on_radioButton_7_toggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 7: _t->on_radioButton_8_toggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 8: _t->on_radioButton_9_toggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 9: _t->on_radioButton_10_toggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 10: _t->timeout(); break;
        case 11: _t->finish(); break;
        default: ;
        }
    }
}

const QMetaObject QCudaCryptoComplex::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_QCudaCryptoComplex.data,
      qt_meta_data_QCudaCryptoComplex,  qt_static_metacall, 0, 0}
};


const QMetaObject *QCudaCryptoComplex::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *QCudaCryptoComplex::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_QCudaCryptoComplex.stringdata))
        return static_cast<void*>(const_cast< QCudaCryptoComplex*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int QCudaCryptoComplex::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 12)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 12;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 12)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 12;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
