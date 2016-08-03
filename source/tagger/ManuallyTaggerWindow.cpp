#include <QApplication>
#include <QMessageBox>
#include <QKeyEvent>
#include <QScrollBar>
#include <QTimer>
#include <QListView>
#include "ManuallyTaggerWindow.h"
#include "utils.h"

using boost::optional;
namespace deeplocalizer {


ManuallyTaggerWindow::ManuallyTaggerWindow(std::unique_ptr<ManuallyTagger> tagger) :
    QMainWindow(nullptr),
    _tagger(std::move(tagger))
{
    init();
}

ManuallyTaggerWindow::ManuallyTaggerWindow(std::vector<ImageDescPtr> && descriptions) :
    QMainWindow(nullptr),
    _tagger(std::make_unique<ManuallyTagger>(std::move(descriptions)))
{
    init();
}

void ManuallyTaggerWindow::init() {
    ui = new Ui::ManuallyTaggerWindow;
    ui->setupUi(this);
    _grid_layout = new QGridLayout(ui->scrollArea);
    _whole_image = new WholeImageWidget(ui->scrollArea);
    _progres_bar = new QProgressBar(ui->statusbar);
    _image_list_model = new QStringListModel(this);
    _save_timer = new QTimer(this);
    _save_timer->start(10000);
    ui->scrollArea->setAlignment(Qt::AlignCenter);
    ui->scrollArea->setWidgetResizable(true);
    setupActions();
    setupConnections();
    setupUi();
    _state = State::Tags;
    _tagger->loadCurrentImage();
}

ManuallyTaggerWindow::~ManuallyTaggerWindow()
{
    save();
    delete ui;
}

void ManuallyTaggerWindow::showImage() {
    _state = State::Image;
    _whole_image->setTags(_image->getCvMat(), &_desc->getTags());
    ui->scrollArea->takeWidget();
    ui->scrollArea->setWidget(_whole_image);
    ui->scrollArea->setBackgroundRole(QPalette::Dark);
    ui->scrollArea->show();
}

void ManuallyTaggerWindow::showTags() {
    _state = State::Tags;
    auto & tags = _desc->getTags();
    if(tags.size() == 0) {
        this->next();
        return;
    }
    _tag_widgets.clear();
    for (auto & tag : tags) {
        auto mat = tag.getSubimage(_image->getCvMat(),
                                   TagWidget::DEFAULT_BORDER);
        TagWidgetPtr widget = std::make_shared<TagWidget>(this, tag, mat);
        connect(widget.get(), &TagWidget::clicked, widget.get(), &TagWidget::toggleTag);
        connect(widget.get(), &TagWidget::clicked, this, &ManuallyTaggerWindow::changed);
        _tag_widgets.push_back(widget);
    }
    arangeTagWidgets();
    if (ui->tags_container->layout() != _grid_layout) {
        delete ui->tags_container->layout();
        ui->tags_container->setLayout(_grid_layout);
    }
    // do not delete current widget in scrollArea
    ui->scrollArea->takeWidget();
    ui->scrollArea->setWidget(ui->tags_container);
    _whole_image->hide();
}

void ManuallyTaggerWindow::arangeTagWidgets() {
    int col = 0;
    int row = 0;
    int width = ui->scrollArea->geometry().width();
    ASSERT(not _tag_widgets.empty(), "_tag_widgets is empty. Got " << _desc->getTags().size() << " Tags.");
    int col_width = (_tag_widgets.front()->width() + _grid_layout->horizontalSpacing());
    int cols = width / col_width;

    for(auto & widget : _tag_widgets) {
        if (col == cols) { col = 0; row++; }
        _grid_layout->addWidget(widget.get(), row, col);
        col++;
    }
}

void ManuallyTaggerWindow::resizeEvent(QResizeEvent * ) {
    if(_state == State::Tags) {
        arangeTagWidgets();
    }
}

void ManuallyTaggerWindow::setupActions() {
    ui->actionZoomIn->setShortcuts({QKeySequence::ZoomIn, Qt::SHIFT + Qt::Key_J});
    ui->actionZoomOut->setShortcuts({QKeySequence::ZoomOut, Qt::SHIFT + Qt::Key_K});
    ui->actionScrollLeft->setShortcuts({Qt::Key_A, Qt::Key_H});
    ui->actionScrollUp->setShortcuts({Qt::Key_W, Qt::Key_K});
    ui->actionScrollRight->setShortcuts({Qt::Key_D, Qt::Key_L});
    ui->actionScrollDown->setShortcuts({Qt::Key_S, Qt::Key_J});

    addAction(ui->actionNext);
    addAction(ui->actionBack);
    addAction(ui->actionScroll);
    addAction(ui->actionScroolBack);
    addAction(ui->actionZoomIn);
    addAction(ui->actionZoomOut);
    addAction(ui->actionSave);
    addAction(ui->actionScrollLeft);
    addAction(ui->actionScrollRight);
    addAction(ui->actionScrollUp);
    addAction(ui->actionScrollDown);

    connect(ui->actionNext, &QAction::triggered, this, &ManuallyTaggerWindow::next);
    connect(ui->actionBack, &QAction::triggered, this, &ManuallyTaggerWindow::back);
    connect(ui->actionNext, &QAction::triggered, this, &ManuallyTaggerWindow::updateStatusBar);
    connect(ui->actionBack, &QAction::triggered, this, &ManuallyTaggerWindow::updateStatusBar);

    connect(ui->actionScroll, &QAction::triggered, this, &ManuallyTaggerWindow::scroll);
    connect(ui->actionScroolBack, &QAction::triggered, this, &ManuallyTaggerWindow::scrollBack);
    connect(ui->actionScrollLeft, &QAction::triggered, this, &ManuallyTaggerWindow::scrollLeft);
    connect(ui->actionScrollRight, &QAction::triggered, this, &ManuallyTaggerWindow::scrollRight);
    connect(ui->actionScrollUp, &QAction::triggered, this, &ManuallyTaggerWindow::scrollTop);
    connect(ui->actionScrollDown, &QAction::triggered, this, &ManuallyTaggerWindow::scrollBottom);

    connect(ui->actionZoomIn, &QAction::triggered, _whole_image, &WholeImageWidget::zoomIn);
    connect(ui->actionZoomOut, &QAction::triggered, _whole_image, &WholeImageWidget::zoomOut);
    connect(ui->actionSave, &QAction::triggered, this, &ManuallyTaggerWindow::save);
}

void ManuallyTaggerWindow::setupConnections() {
    connect(ui->push_next, &QPushButton::clicked, ui->actionNext, &QAction::trigger);
    connect(ui->push_back, &QPushButton::clicked, ui->actionBack, &QAction::trigger);
    connect(_whole_image, &WholeImageWidget::changed, this, &ManuallyTaggerWindow::changed);
    connect(_tagger.get(), &ManuallyTagger::loadedImage, this, &ManuallyTaggerWindow::setImage);
    connect(_tagger.get(), &ManuallyTagger::outOfRange, []() {
        QMessageBox box;
        box.setWindowTitle("DONE!");
        box.setText("You are done! Feel very very happy! :-)");
        box.exec();
    });
    connect(_tagger.get(), &ManuallyTagger::progress, this, &ManuallyTaggerWindow::setProgress);
    connect(_save_timer, &QTimer::timeout, this, [this]() { this->save(false); });
    connect(ui->imagesListView, &QListView::clicked, [this](const QModelIndex & idx) {
        _tagger->loadImage(idx.row());
    });
}

void ManuallyTaggerWindow::setProgress(double progress) {
    int distance = _progres_bar->maximum() - _progres_bar->minimum();
    int value = _progres_bar->minimum() + static_cast<int>(distance * progress);
    _progres_bar->setValue(value);
}

void ManuallyTaggerWindow::updateStatusBar() {
    auto message = tr("#") + QString::number(this->_tagger->getIdx() + 1) + "      " +
                   QString::fromStdString(this->_desc->filename);
    if (_changed) {
        message += " * ";
    }
    this->ui->statusbar->showMessage(message);
}

void ManuallyTaggerWindow::setupUi() {
    ui->statusbar->addPermanentWidget(_progres_bar);
    setProgress(0);
    _image_list_model->setStringList(fileStringList());
    ui->imagesListView->setModel(_image_list_model);

}

QStringList ManuallyTaggerWindow::fileStringList() {
    QStringList list;
    auto & descs = _tagger->getImageDescs();
    for(size_t i = 0; i < descs.size(); i++) {
        const auto &desc = descs.at(i);
        QString check;
        if (_tagger->isDone(i)) {
            check = "\u2713  ";
        }
        list.append("#" + QString::number(i+1) + ":   " + check +
                            QString::fromStdString(desc->filename));
    }
    return list;
}
void ManuallyTaggerWindow::next() {
    if (_state == State::Image) {
        _next_state = State::Tags;
        _tagger->doneTagging();
        _image_list_model->setStringList(fileStringList());
        _tagger->loadNextImage();
    } else if(_state == State::Tags) {
        eraseNegativeTags();
        showImage();
    }
}

void ManuallyTaggerWindow::back() {
    if (_state == State::Image) {
        showTags();
    } else if(_state == State::Tags) {
        _next_state = State::Image;
        _tagger->loadLastImage();
    }
}

void ManuallyTaggerWindow::scroll() {
    QScrollBar * vert = ui->scrollArea->verticalScrollBar();
    QScrollBar * horz = ui->scrollArea->horizontalScrollBar();
    int next_horz = horz->value() + horz->pageStep() / 2;
    int next_vert = vert->value() + static_cast<int>(vert->pageStep() * 0.8);

    if(horz->value() == horz->maximum() && vert->value() == vert->maximum()) {
        next();
    } else if(horz->value() == horz->maximum()) {
        horz->setValue(0);
        if(next_vert > vert->maximum()) {
            vert->setValue(vert->maximum());
        } else {
            vert->setValue(next_vert);
        }
    } else if(next_horz > horz->maximum()) {
        horz->setValue(horz->maximum());
    } else {
        horz->setValue(next_horz);
    }
}

void ManuallyTaggerWindow::scrollBack() {
    QScrollBar * vert = ui->scrollArea->verticalScrollBar();
    QScrollBar * horz = ui->scrollArea->horizontalScrollBar();
    int last_horz = horz->value() - horz->pageStep() / 2;
    int last_vert = vert->value() - static_cast<int>(vert->pageStep() * 0.8);
    if(horz->value() == horz->minimum() && vert->value() == vert->minimum()) {
        back();
    } else if(horz->value() == horz->minimum()) {
        horz->setValue(horz->maximum());
        if(last_vert < vert->minimum()) {
            vert->setValue(vert->minimum());
        } else {
            vert->setValue(last_vert);
        }
    } else if(last_horz < horz->minimum()) {
        horz->setValue(horz->minimum());
    } else {
        horz->setValue(last_horz);
    }
}

void ManuallyTaggerWindow::scrollLeft() {
    auto horz = ui->scrollArea->horizontalScrollBar();
    horz->setValue(horz->value() - horz->pageStep()/8);
}

void ManuallyTaggerWindow::scrollRight() {
    auto horz = ui->scrollArea->horizontalScrollBar();
    horz->setValue(int(horz->value() + horz->pageStep()/8));
}

void ManuallyTaggerWindow::scrollTop() {
    auto vert = ui->scrollArea->verticalScrollBar();
    vert->setValue(int(vert->value() - vert->pageStep()/8));
}

void ManuallyTaggerWindow::scrollBottom() {
    auto vert = ui->scrollArea->verticalScrollBar();
    vert->setValue(int(vert->value() + vert->pageStep()/8));
}

void ManuallyTaggerWindow::setImage(unsigned long idx, ImageDescPtr desc,
                                    ImagePtr img) {
    _desc = desc;
    _image = img;
    updateStatusBar();
    ui->imagesListView->setCurrentIndex(_image_list_model->index(idx));
    if (_next_state == State::Image) {
        showImage();
    } else {
        showTags();
    }

}

void ManuallyTaggerWindow::eraseNegativeTags() {
    auto & tags = _desc->getTags();
    tags.erase(std::remove_if(tags.begin(), tags.end(), [](const auto & tag) {
        return tag.isNoTag();
    }), tags.end());
}

void ManuallyTaggerWindow::changed() {
    _changed = true;
    updateStatusBar();
}
void ManuallyTaggerWindow::save(bool all_descs) {
    if(_changed) {
        _tagger->save();
        if (all_descs) {
        } else {
            if (_desc) {
                _desc->save();
            }
        }
        _changed = false;
        updateStatusBar();
    }
}
}
