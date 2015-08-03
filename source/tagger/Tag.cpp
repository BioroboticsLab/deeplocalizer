
#include "Tag.h"

#include <QPainter>
#include <mutex>

namespace deeplocalizer {


using boost::optional;


cv::Rect centerBoxAtEllipse(const cv::Rect & bb,
                            const pipeline::Ellipse & ellipse) {
    cv::Point2i center = ellipse.getCen();
    cv::Rect box(bb.x + center.x - TAG_WIDTH / 2,
                 bb.y + center.y - TAG_HEIGHT / 2,
                 TAG_WIDTH, TAG_HEIGHT);
    return box;
}

cv::Rect centerBox(const cv::Rect & bb) {
    cv::Point center(bb.x + bb.width / 2, bb.y + bb.height / 2);
    cv::Rect box(center.x - TAG_WIDTH / 2,
                 center.y - TAG_HEIGHT / 2,
                 TAG_WIDTH, TAG_HEIGHT);
    return box;
}

std::atomic_long Tag::id_counter(0);

unsigned long Tag::generateId() {
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    static thread_local std::uniform_int_distribution<unsigned long> dis(0, ULONG_MAX);
    unsigned  long id = dis(gen);
    return id;
}

Tag::Tag() :  _id(Tag::generateId())
{
}

Tag::Tag(const pipeline::Tag & pipetag) :  _id(Tag::generateId()) {
    boost::optional<pipeline::Ellipse> ellipse;
    for(auto candidate : pipetag.getCandidatesConst()) {
        if (!ellipse.is_initialized()) {
            ellipse = optional<pipeline::Ellipse>(candidate.getEllipse());
        } else if (ellipse.get() < candidate.getEllipse()) {
            ellipse = optional<pipeline::Ellipse>(candidate.getEllipse());
        }
    }
    if(ellipse) {
        _boundingBox = centerBoxAtEllipse(pipetag.getBox(), ellipse.get());
        _ellipse = ellipse;
        _ellipse.get().setCen(cv::Point2i(TAG_WIDTH/2, TAG_HEIGHT/2));
    } else {
        _boundingBox = centerBox(pipetag.getBox());
        _is_tag = IsTag::No;
    }
}

Tag::Tag(cv::Rect boundingBox) :
    _id(Tag::generateId()),
    _boundingBox(std::move(boundingBox)),
    _ellipse(optional<pipeline::Ellipse>())
{
}

Tag::Tag(cv::Rect boundingBox, optional<pipeline::Ellipse> ellipse) :
        _id(Tag::generateId()),
        _boundingBox(boundingBox),
        _ellipse(ellipse)
{
}

void Tag::guessIsTag(int threshold) {
    if(_ellipse.is_initialized() && _ellipse.get().getVote() > threshold) {
        _is_tag = IsTag::Yes;
    } else {
        _is_tag = IsTag::No;
    }
}
const optional<pipeline::Ellipse> & Tag::getEllipse () const {
    return _ellipse;
}
const cv::Rect & Tag::getBoundingBox() const {
    return _boundingBox;
}

void Tag::setBoundingBox(cv::Rect rect) {
    _boundingBox = rect;
}

IsTag Tag::isTag() const {
    return _is_tag;
}

void Tag::setIsTag(IsTag isTag) {
        this->_is_tag = isTag;
}

void Tag::toggleIsTag() {
    if (_is_tag == IsTag::Yes) {
        _is_tag = IsTag::No;
    } else if (_is_tag == IsTag::No) {
        _is_tag = IsTag::Yes;
    }
}

bool Tag::operator==(const Tag &other) const {
    if (_ellipse.is_initialized() && other._ellipse.is_initialized()) {
        auto te = _ellipse.get();
        auto oe = other._ellipse.get();
        return (_boundingBox == other._boundingBox &&
                _is_tag == other._is_tag &&
                te.getAngle() == oe.getAngle() &&
                te.getVote() == oe.getVote() &&
                te.getAxis() == oe.getAxis() &&
                te.getCen() == oe.getCen());
    }
    return _ellipse.is_initialized() == other._ellipse.is_initialized();
}


cv::Mat Tag::getSubimage(const cv::Mat & orginal, unsigned int border) const {
    cv::Rect box = _boundingBox;
    box.x -= border;
    box.y -= border;
    box.width += 2*border;
    box.height+= 2*border;
    if(box.x < 0) box.x = 0;
    if(box.y < 0) box.y = 0;
    if(box.width + box.x >= orginal.cols) box.x = orginal.cols - box.width - 1;
    if(box.height + box.y >= orginal.rows) box.y = orginal.rows - box.height - 1;

    return orginal(box).clone();
}

void Tag::draw(QPainter & p, int lineWidth) const {
    auto bb = _boundingBox;
    if (_is_tag == IsTag::Yes) {
        p.setPen(QPen(Qt::green, lineWidth));
    } else if (_is_tag == IsTag::No) {
        p.setPen(QPen(Qt::red, lineWidth));
    } else if (_is_tag == IsTag::Exclude) {
        p.setPen(QPen(Qt::magenta, lineWidth));
    }
    p.drawRect(QRect(bb.x, bb.y, bb.height, bb.width));
}
void Tag::drawEllipse(QPainter & p, int lineWidth, bool drawVote) const {
    static const QPoint zero(0, 0);
    if(not _ellipse) {
        return;
    }
    auto e = _ellipse.get();
    auto bb = _boundingBox;
    p.setPen(Qt::blue);
    QPoint center(bb.x+e.getCen().x, bb.y+e.getCen().y);
    QFont font = p.font();
    font.setPointSizeF(18);
    p.setPen(Qt::blue);
    if(drawVote) {
        p.setFont(font);
        p.drawText(bb.x - 10 , bb.y - 10, QString::number(e.getVote()));
    }
    p.setPen(QPen(Qt::blue, lineWidth));
    p.save();
    p.translate(center);
    p.rotate(e.getAngle());
    p.drawLine(-4, 0, 4, 0);
    p.drawLine(0, 4, 0, -4);
    p.drawEllipse(zero, e.getAxis().width, e.getAxis().height);
    p.restore();
}

unsigned long Tag::id() const {
    return _id;
}
void Tag::setId(unsigned long id) {
    _id = id;
}
}


