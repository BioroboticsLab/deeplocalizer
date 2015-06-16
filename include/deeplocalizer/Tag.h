
#ifndef DEEP_LOCALIZER_TAG_H
#define DEEP_LOCALIZER_TAG_H

#include <atomic>

#include <QMetaType>
#include <QString>

#include <opencv2/core/core.hpp>

#include "pipeline/datastructure/Tag.h"
#include "pipeline/datastructure/Ellipse.h"
#include "pipeline/datastructure/serialization.hpp"
#include "serialization.h"

class QPainter;

namespace deeplocalizer {
namespace  tagger {

const int TAG_WIDTH = 100;
const int TAG_HEIGHT = 100;


enum IsTag{
    Yes,
    No,
    Exclude
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

    IsTag isTag() const;
    void setIsTag(IsTag is_tag);
    void toggleIsTag();

    bool isExclude() const {
        return _is_tag == Exclude;
    }
    bool isYes() const {
        return _is_tag == Yes;
    }
    bool isNo() const {
        return _is_tag == No;
    }

    cv::Point2i center() const {
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

private:
    unsigned long _id;
    cv::Rect _boundingBox;
    boost::optional<pipeline::Ellipse> _ellipse;
    IsTag _is_tag = IsTag::Yes;

    static unsigned long generateId();
    static std::atomic_long id_counter;

    friend class boost::serialization::access;
    template<class Archive>
    void save(Archive & ar, const unsigned int) const
    {
        ar & BOOST_SERIALIZATION_NVP(_boundingBox);
        ar & BOOST_SERIALIZATION_NVP(_ellipse);
        ar & BOOST_SERIALIZATION_NVP(_is_tag);

    }
    template<class Archive>
    void load(Archive & ar, const unsigned int)
    {
        ar & BOOST_SERIALIZATION_NVP(_boundingBox);
        ar & BOOST_SERIALIZATION_NVP(_ellipse);
        ar & BOOST_SERIALIZATION_NVP(_is_tag);
        auto center = this->center();
        _boundingBox = cv::Rect(center.x - TAG_WIDTH/2, center.y - TAG_WIDTH/2,
                                TAG_WIDTH, TAG_HEIGHT);
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};
}
}

Q_DECLARE_METATYPE(deeplocalizer::tagger::Tag)

#endif //DEEP_LOCALIZER_TAG_H
