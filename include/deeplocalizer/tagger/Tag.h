
#ifndef DEEP_LOCALIZER_TAG_H
#define DEEP_LOCALIZER_TAG_H

#include <atomic>
#include <random>

#include <QMetaType>
#include <QString>

#include <opencv2/core/core.hpp>
#include <json.hpp>


#include "deeplocalizer_tagger.h"

class QPainter;

namespace deeplocalizer {


enum TagType {
    IsTag,
    NoTag,
    Exclude,
    BeeWithoutTag,
};

class Tag {
public:
    Tag();
    Tag(cv::Rect boundingBox);
    static const int IS_TAG_THRESHOLD = 1200;

    unsigned long id() const;
    void setId(unsigned long id);
    const cv::Rect & getBoundingBox() const;
    void setBoundingBox(cv::Rect boundingBox);

    TagType type() const;
    void setType(TagType tagtype);
    void toggleIsTag();

    inline bool isExclude() const {
        return _tag_type == Exclude;
    }
    inline bool isTag() const {
        return _tag_type == IsTag;
    }
    inline bool isNoTag() const {
        return _tag_type == NoTag;
    }
    inline bool isBeeWithoutTag() const {
        return _tag_type == BeeWithoutTag;
    }

    inline cv::Point2i center() const {
        return cv::Point2i{
                _boundingBox.x + _boundingBox.width  / 2,
                _boundingBox.y + _boundingBox.height / 2
        };
    }

    cv::Mat getSubimage(const cv::Mat &orginal, unsigned int border=0) const;
    bool operator==(const Tag &other) const;
    void guessIsTag(int threshold = IS_TAG_THRESHOLD);
    void draw(QPainter & p, int lineWidth = 3) const;

    nlohmann::json to_json() const;
    static Tag from_json(const nlohmann::json &);

private:
    unsigned long _id;
    cv::Rect _boundingBox;
    TagType _tag_type = TagType::IsTag;

    static unsigned long generateId();
    static std::atomic_long id_counter;
};
}

Q_DECLARE_METATYPE(deeplocalizer::Tag)

#endif //DEEP_LOCALIZER_TAG_H
