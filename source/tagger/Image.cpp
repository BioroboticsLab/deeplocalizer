
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/filesystem.hpp>
#include <boost/optional.hpp>

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
    this->tags.emplace_back(tag);
}

void ImageDesc::addTag(Tag tag) {
    this->tags.emplace_back(std::move(tag));
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
        std::cout << desc.savePath() << " exists: " <<  io::exists(desc.savePath()) << std::endl;
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
    safe_serialization(path, this->to_json());
}

ImageDescPtr ImageDesc::load(const std::string & path) {
    std::ifstream is(path);
    nlohmann::json json_desc;
    is >>  json_desc;
    return std::make_shared<ImageDesc>(ImageDesc::from_json(json_desc));
}

Image::Image() {
}

Image::Image(const ImageDesc & descr) : _filename(descr.filename)  {
    ASSERT(io::exists(_filename), "Cannot open file: " << _filename);
    _mat = cv::imread(_filename, cv::IMREAD_GRAYSCALE);
}


bool Image::write(const io::path & path, boost::optional<std::pair<int, int>> compression) const {
    io::path p;
    if (path.empty()) {
        p = _filename;
    } else {
        p = path;
    }
    std::vector<int> compress_vec;
    if (compression) {
        compress_vec.push_back(std::get<0>(compression.get()));
        compress_vec.push_back(std::get<1>(compression.get()));
    }
    return cv::imwrite(p.string(), _mat, compress_vec);
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

using json = nlohmann::json;
json ImageDesc::to_json() const {
    json j;
    j["filename"] = this->filename;
    j["tags"] = json::array();
    for(const auto & tag : this->tags) {
        j["tags"].push_back(tag.to_json());
    }
    return j;
}

ImageDesc ImageDesc::from_json(const json &j) {
    std::vector<Tag> tags;
    for(const auto & jtag : j["tags"]) {
        tags.push_back(Tag::from_json((jtag)));
    }
    return ImageDesc(j["filename"], tags);
}

}
