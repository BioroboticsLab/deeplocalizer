#ifndef MANUELLTAGWINDOW_H
#define MANUELLTAGWINDOW_H

#include <QMainWindow>
#include <QProgressBar>
#include <QStringListModel>

#include "ui_ManuallyTaggerWindow.h"
#include "ManuallyTagger.h"
#include "WholeImageWidget.h"

namespace deeplocalizer {


class ManuallyTaggerWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit ManuallyTaggerWindow(std::vector<ImageDescPtr> && _image_desc);
    explicit ManuallyTaggerWindow(std::unique_ptr<ManuallyTagger> tagger);
    ~ManuallyTaggerWindow();
public slots:
    void next();
    void back();
    void scrollLeft();
    void scrollRight();
    void scrollTop();
    void scrollBottom();
    void scroll();
    void scrollBack();
    void changed();
    void save(bool all_descs=false);
    void setImage(unsigned long idx, ImageDescPtr desc,
                                        ImagePtr img);
private slots:
    void updateStatusBar();
    void setProgress(double progress);
private:
    Ui::ManuallyTaggerWindow *ui;

    QGridLayout * _grid_layout;
    WholeImageWidget * _whole_image;
    QProgressBar * _progres_bar;
    QStringListModel *_image_list_model;

    std::unique_ptr<ManuallyTagger> _tagger;
    ImageDescPtr  _desc;
    ImagePtr  _image;
    QTimer * _save_timer;
    bool _changed = false;


    void init();
    void showImage();
    void setupConnections();
    void setupActions();
    void setupUi();
    void eraseNegativeTags();
    QStringList fileStringList();
};
}
#endif // MANUELLTAGWINDOW_H
