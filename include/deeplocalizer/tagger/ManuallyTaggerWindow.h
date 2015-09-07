#ifndef MANUELLTAGWINDOW_H
#define MANUELLTAGWINDOW_H

#include <QMainWindow>
#include <QProgressBar>
#include <QStringListModel>

#include "ui_ManuallyTaggerWindow.h"
#include "ManuallyTagger.h"
#include "WholeImageWidget.h"
#include "TagWidget.h"


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
protected:
    void resizeEvent(QResizeEvent * );
private slots:
    void updateStatusBar();
    void setProgress(double progress);
private:
    enum class State{Tags, Image};
    Ui::ManuallyTaggerWindow *ui;

    QGridLayout * _grid_layout;
    WholeImageWidget * _whole_image;
    std::vector<TagWidgetPtr> _tag_widgets;
    QProgressBar * _progres_bar;
    QStringListModel *_image_list_model;

    std::unique_ptr<ManuallyTagger> _tagger;
    State _state = State::Tags;
    State _next_state = State::Tags;
    ImageDescPtr  _desc;
    ImagePtr  _image;
    QTimer * _save_timer;
    bool _changed = false;


    void init();
    void showImage();
    void showTags();
    void arangeTagWidgets();
    void setupConnections();
    void setupActions();
    void setupUi();
    void eraseNegativeTags();
    QStringList fileStringList();
};
}
#endif // MANUELLTAGWINDOW_H
