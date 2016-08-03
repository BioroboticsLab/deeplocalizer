
#include "Tag.h"

#include <QPainter>
#include <mutex>
#include <boost/optional.hpp>

namespace deeplocalizer {


using boost::optional;


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

Tag::Tag(cv::Rect boundingBox) :
        _id(Tag::generateId()),
        _boundingBox(boundingBox),
        _tag_type(IsTag) { }

const cv::Rect & Tag::getBoundingBox() const {
    return _boundingBox;
}

void Tag::setBoundingBox(cv::Rect rect) {
    _boundingBox = rect;
}

TagType Tag::type() const {
    return _tag_type;
}

void Tag::setType(TagType tagtype) {
        this->_tag_type = tagtype;
}

void Tag::toggleIsTag() {
    if (_tag_type == TagType::IsTag) {
        _tag_type = TagType::NoTag;
    } else if (_tag_type == TagType::NoTag) {
        _tag_type = TagType::IsTag;
    }
}

bool Tag::operator==(const Tag &other) const {
    return _boundingBox == other._boundingBox &&
            _tag_type == other._tag_type;
}


cv::Mat Tag::getSubimage(const cv::Mat & orginal, unsigned int border) const {
    return ::deeplocalizer::getSubimage(orginal, _boundingBox, border);
}

void Tag::draw(QPainter & p, int lineWidth) const {
    auto bb = _boundingBox;
    auto set_pen = [&](double width) {
        if (isTag()) {
            p.setPen(QPen(Qt::green, width));
        } else if (isNoTag()) {
            p.setPen(QPen(Qt::red, width));
        } else if (isExclude()) {
            p.setPen(QPen(Qt::magenta, width));
        } else if (isBeeWithoutTag()) {
            p.setPen(QPen(Qt::cyan, width));
        }
    };
    set_pen(lineWidth);
    p.drawArc(QRect(bb.x, bb.y, bb.height, bb.width), 0, 16*360);
    auto c = center();
    set_pen(1);
    int o = 3;
    p.drawLine(c.x - o, c.y - o, c.x + o, c.y + o);
    p.drawLine(c.x - o, c.y + o, c.x + o, c.y - o);
}

unsigned long Tag::id() const {
    return _id;
}
void Tag::setId(unsigned long id) {
    _id = id;
}

using json = nlohmann::json;

std::string tagtype_to_string(TagType tagType) {
    switch (tagType) {
        case TagType::NoTag:
            return "notag";
        case TagType::BeeWithoutTag:
            return "bee_without_tag";
        case TagType::Exclude:
            return "exclude";
        case TagType::IsTag:
            return "istag";
        default:
            throw "unknown tag type";
    }
}

TagType tagtype_from_string(const std::string & str) {
    if (str == "notag") {
        return TagType::NoTag;
    } else if (str == "bee_without_tag") {
        return TagType::BeeWithoutTag;
    } else if (str == "exclude") {
        return TagType::Exclude;
    } else if (str == "istag") {
        return TagType::IsTag;
    } else {
        throw "unknown tag type";
    }
}

json Tag::to_json() const {
    json jtag;
    jtag["x"] = this->center().x;
    jtag["y"] = this->center().y;
    jtag["tagtype"] = tagtype_to_string(_tag_type);
    return jtag;
}

Tag Tag::from_json(const json &j) {
    cv::Point2i center(j["x"], j["y"]);
    cv::Rect boundingBox(center.x - TAG_WIDTH/2, center.y - TAG_WIDTH/2,
                           TAG_WIDTH, TAG_HEIGHT);
    Tag tag(boundingBox);
    tag.setType(tagtype_from_string(j["tagtype"]));
    return tag;
}
}


