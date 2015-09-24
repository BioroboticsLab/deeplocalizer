#ifndef DEEP_LOCALIZER_IMAGE_H
#define DEEP_LOCALIZER_IMAGE_H

#include <boost/optional.hpp>
#include <boost/serialization/list.hpp>
#include <boost/filesystem.hpp>

#include "Tag.h"
#include "serialization.h"

class QPixmap;

namespace deeplocalizer {

class ImageDesc {
public:
    std::string filename;
    ImageDesc();
    ImageDesc(const std::string filename);
    ImageDesc(const std::string filename, std::vector<Tag> _tags);
    QPixmap visualise_tags();
    void addTag(Tag&& tag);
    void setTags(std::vector<Tag> && tag);
    const std::vector<Tag> & getTags() const;
    std::vector<Tag> & getTags();
    bool operator==(const ImageDesc & other) const;
    void save();
    void save(const std::string &path);
    void setSavePathExtension(std::string ext);
    std::string savePath() const;
    static std::shared_ptr<ImageDesc> load(const std::string &path);
    static std::vector<ImageDesc> fromPathFile(const std::string &path,
                                               const std::string & image_desc_extension = "desc");
    static std::vector<ImageDesc> fromPaths(const std::vector<std::string> paths,
                                            const std::string & image_desc_extension = "desc");

    static std::vector<std::shared_ptr<ImageDesc>> fromPathsPtr(const std::vector<std::string> paths,
                                                             const std::string & image_desc_extension);
    static std::vector<std::shared_ptr<ImageDesc>> fromPathFilePtr(
            const std::string &path, const std::string & image_desc_extension = "desc");

private:
    std::string _save_extension = ".desc";
    std::vector<Tag> tags;
    unsigned long current_tag = 0;

    friend class boost::serialization::access;
    template <class Archive>
    void serialize( Archive & ar, const unsigned int)
    {
      ar & BOOST_SERIALIZATION_NVP(filename);
      ar & BOOST_SERIALIZATION_NVP(tags);
    }
};
using ImageDescPtr = std::shared_ptr<ImageDesc>;

class Image {
public:
    explicit Image();
    explicit Image(const ImageDesc & descr);
    cv::Mat getCvMat() const;
    void beesBookPreprocess(bool use_hist_eq);
    bool write(boost::filesystem::path path = {}) const;
    bool operator==(const Image & other) const;
    const std::string & filename() const {
        return _filename;
    }
    void applyLocalHistogramEq();
private:
    cv::Mat _mat;
    std::string _filename;
};

using ImagePtr = std::shared_ptr<Image>;

}

Q_DECLARE_METATYPE(deeplocalizer::ImageDesc)

#endif //DEEP_LOCALIZER_IMAGE_H
