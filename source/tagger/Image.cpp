
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/filesystem.hpp>
#include <boost/optional.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/deque.hpp>
#include "serialization.h"

#include "Image.h"
#include "Tag.h"
#include "utils.h"
#include "qt_helper.h"

namespace deeplocalizer {


using namespace std;
namespace io = boost::filesystem;

ImageDesc::ImageDesc() {

}

ImageDesc::ImageDesc(const std::string _filename) : filename(_filename) {

}

ImageDesc::ImageDesc(const std::string _filename, std::vector<Tag> _tags) :
        filename(_filename), tags(_tags) {

}

const std::vector<Tag> &ImageDesc::getTags() const {
    return tags;
}
std::vector<Tag> &ImageDesc::getTags() {
    return tags;
}

QPixmap ImageDesc::visualise_tags() {
    return QPixmap();
}

void ImageDesc::addTag(Tag && tag) {
    this->tags.push_back(tag);
}

void ImageDesc::setTags(std::vector<Tag> && tags) {
    this->tags = tags;
}


bool ImageDesc::operator==(const ImageDesc & other) const {
    return (filename == other.filename &&
            tags == other.tags);
}

std::vector<ImageDescPtr> ImageDesc::fromPathFilePtr(const std::string &path,
                                               const std::string & image_desc_extension) {
    auto image_descs = fromPathFile(path, image_desc_extension);
    std::vector<ImageDescPtr> image_desc_ptrs;
    for(auto & desc : image_descs) {
        image_desc_ptrs.emplace_back(std::make_shared<ImageDesc>(std::move(desc)));
    }
    return image_desc_ptrs;
}

std::vector<ImageDesc> ImageDesc::fromPathFile(const std::string &path,
                                               const std::string & image_desc_extension) {
    const io::path pathfile(path);
    ASSERT(io::exists(pathfile), "File " << pathfile << " does not exists.");
    ifstream ifs{pathfile.string()};
    std::string path_to_image;
    std::vector<std::string> paths;
    for(int i = 0; std::getline(ifs, path_to_image); i++) {
        ASSERT(io::exists(path_to_image), "File " << path_to_image << " does not exists.");
        paths.push_back(path_to_image);
    }
    return fromPaths(paths, image_desc_extension);
}

std::vector<ImageDescPtr> ImageDesc::fromPathsPtr(const std::vector<std::string> paths,
                                                  const std::string & image_desc_extension) {
    auto image_descs = fromPaths(paths, image_desc_extension);
    std::vector<ImageDescPtr> image_desc_ptrs;
    for(auto & desc : image_descs) {
        image_desc_ptrs.emplace_back(std::make_shared<ImageDesc>(std::move(desc)));
    }
    return image_desc_ptrs;
}

std::vector<ImageDesc> ImageDesc::fromPaths(const std::vector<std::string> paths,
                                            const std::string & image_desc_extension) {
    std::vector<ImageDesc> descs;
    for(auto & path: paths) {
        ASSERT(io::exists(path), "File " << path << " does not exists.");
        auto desc = ImageDesc(path);
        desc.setSavePathExtension(image_desc_extension);
        if(io::exists(desc.savePath())) {
            desc = *ImageDesc::load(desc.savePath());
            desc.filename = path;
        }
        descs.push_back(desc);
    }
    return descs;
}

void ImageDesc::setSavePathExtension(std::string ext) {
    _save_extension = ext;
}

std::string ImageDesc::savePath() const {
    std::string str = filename;
    str.append("." + _save_extension);
    return str;
}
void ImageDesc::save() {
    save(savePath());
}
void ImageDesc::save(const std::string & path) {
    safe_serialization(path, boost::serialization::make_nvp("image_desc", *this));
}

ImageDescPtr ImageDesc::load(const std::string & path) {
    std::ifstream is(path);
    boost::archive::binary_iarchive archive(is);
    ImageDesc img_desc;
    archive >> boost::serialization::make_nvp("image_desc", img_desc);
    return std::make_shared<ImageDesc>(std::move(img_desc));
}

Image::Image() {
}

Image::Image(const ImageDesc & descr) : _filename(descr.filename)  {
    ASSERT(io::exists(_filename), "Cannot open file: " << _filename);
    _mat = cv::imread(_filename, cv::IMREAD_GRAYSCALE);
}

cv::Mat applyClahe(cv::Mat & mat)
{
    cv::Mat out_mat;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4.);
    clahe->setTilesGridSize(cv::Size(16, 16));
    clahe->apply(mat,  out_mat);
    return out_mat;
}

void Image::beesBookPreprocess(bool use_hist_eq) {
    applyLocalHistogramEq();
    auto mat_with_border = cv::Mat(_mat.rows + TAG_HEIGHT,
                                   _mat.cols + TAG_WIDTH, CV_8U);
    cv::copyMakeBorder(_mat, mat_with_border,
                       TAG_HEIGHT / 2, TAG_HEIGHT / 2,
                       TAG_WIDTH  / 2, TAG_WIDTH  / 2,
                       cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
    _mat.release();
    _mat = mat_with_border;
}

cv::Mat Image::getCvMat() const {
    return _mat;
}

bool Image::write(io::path path) const {
    if (path.empty()) {
        return cv::imwrite(_filename, _mat);
    } else {
        return cv::imwrite(path.string(), _mat);
    }
}

bool Image::operator==(const Image &other) const {
    if(_filename != other._filename) {
        return false;
    }
    const cv::Mat & m = _mat;
    const cv::Mat & o = other._mat;

    if (m.empty() && o.empty()) {
        return true;
    }
    if (m.cols != o.cols || m.rows != o.rows ||
            m.dims != o.dims) {
        return false;
    }
    return std::equal(m.begin<uchar>(), m.end<uchar>(), o.begin<uchar>(), o.end<uchar>());
}

void Image::applyLocalHistogramEq()
{
    static const int clip_limit = 4;
    static const cv::Size tile_size(deeplocalizer::TAG_WIDTH, deeplocalizer::TAG_HEIGHT);

    auto clahe = cv::createCLAHE(clip_limit, tile_size);
    cv::Mat image_clahe;
    clahe->apply(_mat, image_clahe);
    _mat = image_clahe;
}
}
