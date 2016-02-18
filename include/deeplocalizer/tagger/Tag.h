
#ifndef DEEP_LOCALIZER_TAG_H
#define DEEP_LOCALIZER_TAG_H

#include <atomic>

#include <QMetaType>
#include <QString>

#include <opencv2/core/core.hpp>
#include <json.hpp>

#include "pipeline/datastructure/Tag.h"
#include "pipeline/datastructure/Ellipse.h"
#include "pipeline/datastructure/serialization.hpp"
#include "serialization.h"
class QPainter;

namespace deeplocalizer {

const int TAG_WIDTH = 60;
const int TAG_HEIGHT = 60;
const cv::Point2i TAG_CENTER{TAG_WIDTH / 2, TAG_HEIGHT / 2};
const cv::Size2i TAG_SIZE{TAG_WIDTH, TAG_HEIGHT};

enum TagType {
    IsTag,
    NoTag,
    Exclude,
    BeeWithoutTag,
};

class Tag {
public:
    Tag();
    Tag(const pipeline::Tag & pipetag);
    Tag(cv::Rect boundingBox);
    Tag(cv::Rect boundingBox, boost::optional<pipeline::Ellipse> ellipse);
    static const int IS_TAG_THRESHOLD = 1200;

    unsigned long id() const;
    void setId(unsigned long id);
    const cv::Rect & getBoundingBox() const;
    void setBoundingBox(cv::Rect boundingBox);

    const boost::optional<pipeline::Ellipse> & getEllipse () const;

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
    void drawEllipse(QPainter & p, int lineWidth = 3,
                          bool drawVote = true) const;

    nlohmann::json to_json() const;
    static Tag from_json(const nlohmann::json &);

private:
    unsigned long _id;
    cv::Rect _boundingBox;
    boost::optional<pipeline::Ellipse> _ellipse;
    TagType _tag_type = TagType::IsTag;

    static unsigned long generateId();
    static std::atomic_long id_counter;

    friend class boost::serialization::access;
    template<class Archive>
    void save(Archive & ar, const unsigned int) const
    {
        ar & BOOST_SERIALIZATION_NVP(_boundingBox);
        ar & BOOST_SERIALIZATION_NVP(_ellipse);
        ar & BOOST_SERIALIZATION_NVP(_tag_type);

    }
    template<class Archive>
    void load(Archive & ar, const unsigned int)
    {
        ar & BOOST_SERIALIZATION_NVP(_boundingBox);
        ar & BOOST_SERIALIZATION_NVP(_ellipse);
        ar & BOOST_SERIALIZATION_NVP(_tag_type);
        auto center = this->center();
        if(_ellipse) {
            _ellipse->setCen(TAG_CENTER);
        }
        _boundingBox = cv::Rect(center.x - TAG_WIDTH/2, center.y - TAG_WIDTH/2,
                                TAG_WIDTH, TAG_HEIGHT);
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};
}

Q_DECLARE_METATYPE(deeplocalizer::Tag)

#endif //DEEP_LOCALIZER_TAG_H
