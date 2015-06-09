
#include "ManuallyTagger.h"

#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>

#include "pipeline/Localizer.h"
#include "serialization.h"
#include "utils.h"

namespace deeplocalizer {
namespace tagger {

using boost::optional;
namespace io = boost::filesystem;


const std::string ManuallyTagger::IMAGE_DESC_EXT = "tagger.desc";
const std::string ManuallyTagger::DEFAULT_SAVE_PATH = "tagger_progress.binary";


ManuallyTagger::ManuallyTagger() {
    init();
}

ManuallyTagger::ManuallyTagger(const std::vector<ImageDesc> & descriptions,
                               const std::string & save_path) :
    _save_path(save_path)
{
    for(auto & descr : descriptions ) {
        _image_descs.push_back(std::make_shared<ImageDesc>(descr));
    }
    init();
}

ManuallyTagger::ManuallyTagger(const std::vector<ImageDescPtr> & descriptions,
                               const std::string & save_path) :
    _save_path(save_path)
{
    for(auto & descr : descriptions ) {
        _image_descs.push_back(descr);
    }
    init();
}

ManuallyTagger::ManuallyTagger(std::vector<ImageDescPtr> && descriptions,
                               const std::string & save_path) :
    _image_descs(std::move(descriptions)),
    _save_path(save_path)
{
    init();
}

void ManuallyTagger::init() {
    for(auto & descr : _image_descs) {
        ASSERT(io::exists(descr->filename),
               "Could not open file " << descr->filename);
        descr->setSavePathExtension(IMAGE_DESC_EXT);
        if (io::exists(descr->savePath())) {
            descr = ImageDesc::load(descr->savePath());
        }
    }

    if (_image_descs.size() != _done_tagging.size()) {
        _done_tagging = std::vector<bool>(_image_descs.size(), true);
        _n_done = 0;
    } else {
        _n_done = std::count(_done_tagging.cbegin(), _done_tagging.cend(), true);
    }
}

void ManuallyTagger::save() const {
    save(savePath());
}
void ManuallyTagger::save(const std::string & path) const {
    safe_serialization(path, boost::serialization::make_nvp("tagger", *this));
}

std::unique_ptr<ManuallyTagger> ManuallyTagger::load(const std::string & path) {
    std::ifstream is(path);
    boost::archive::binary_iarchive archive(is);
    ManuallyTagger * tagger;
    archive >> boost::serialization::make_nvp("tagger", tagger);
    return std::unique_ptr<ManuallyTagger>(tagger);
}
void ManuallyTagger::loadNextImage() {
    loadImage(_image_idx + 1);
}

void ManuallyTagger::loadLastImage() {
    if (_image_idx == 0) { return; }
    loadImage(_image_idx - 1);
}

void ManuallyTagger::loadImage(unsigned long idx) {
    if (idx >= _image_descs.size()) {
        emit outOfRange(idx);
        return;
    }
    _image_idx = idx;
    _image = std::make_shared<Image>(*_image_descs.at(_image_idx));
    _desc = _image_descs.at(_image_idx);
    emit loadedImage(_desc, _image);
    if (_image_idx == 0) { emit firstImage(); }
    if (_image_idx + 1 == _image_descs.size()) { emit lastImage(); }
}
void ManuallyTagger::doneTagging() {
    doneTagging(_image_idx);
}

void ManuallyTagger::doneTagging(unsigned long idx) {
    if (_done_tagging.size() <= idx) {
        qWarning() << "[doneTagging] index " << idx << " exceeded size of images "
            << _done_tagging.size();
    }
    _image_descs.at(idx)->save();
    save();
    _done_tagging.at(idx) = true;
    _n_done++;
    emit progress(static_cast<double>(_n_done)/_image_descs.size());
}
}
}
