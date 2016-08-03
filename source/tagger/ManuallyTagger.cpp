
#include "ManuallyTagger.h"

#include <QDebug>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>

#include "utils.h"
#include "qt_helper.h"

namespace deeplocalizer {


using boost::optional;
namespace io = boost::filesystem;


const std::string ManuallyTagger::IMAGE_DESC_EXT = "tagger.json";
const std::string ManuallyTagger::DEFAULT_SAVE_PATH = "tagger_progress.json";


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
    if(_loaded_from_serialization) {
        _image_descs = ImageDesc::fromPathsPtr(_image_paths, IMAGE_DESC_EXT);
    }
    _image_paths.clear();
    for(auto & descr : _image_descs) {
        ASSERT(io::exists(descr->filename),
               "Could not open file " << descr->filename);
        descr->setSavePathExtension(IMAGE_DESC_EXT);
    }
    // sort images that are allready done to the begining
    std::sort(_image_descs.begin(), _image_descs.end(),
              [this](const ImageDescPtr &d1, const ImageDescPtr &d2){
                    return isDone(*d1) > isDone(*d2);
    });
    for(auto & descr : _image_descs) {
        if (io::exists(descr->savePath())) {
            descr = ImageDesc::load(descr->savePath());
        } else {
            descr->setSavePathExtension(".proposal.json");
            if (io::exists(descr->savePath())) {
                descr = ImageDesc::load(descr->savePath());
            }
        }
        descr->setSavePathExtension(IMAGE_DESC_EXT);
        _image_paths.push_back(descr->filename);
    }
    if (_image_descs.size() != _done_tagging.size()) {
        _done_tagging = std::vector<bool>(_image_descs.size(), false);
        _n_done = 0;
    } else {
        _n_done = std::count(_done_tagging.cbegin(), _done_tagging.cend(), true);
    }
}

void ManuallyTagger::save(bool all_descs) const {
    save(savePath());
    if (all_descs) {
        for(auto & desc: _image_descs) {
            desc->save();
        }
    } else {
        _image_descs.at(_image_idx)->save();
    }

}
void ManuallyTagger::save(const std::string & path) const {
    safe_serialization(path, this->to_json());
}

std::unique_ptr<ManuallyTagger> ManuallyTagger::load(const std::string & path) {
    std::ifstream is(path);
    nlohmann::json j;
    is >> j;
    auto tagger = ManuallyTagger::from_json(j);
    tagger->_loaded_from_serialization = true;
    tagger->init();
    return tagger;

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
    emit loadedImage(_image_idx, _desc, _image);
    if (_image_idx == 0) { emit firstImage(); }
    if (_image_idx + 1 == _image_descs.size()) { emit lastImage(); }
}

void ManuallyTagger::loadCurrentImage() {
    loadImage(_image_idx);
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

nlohmann::json ManuallyTagger::to_json() const {
    nlohmann::json j;
    j["image_idx"] = _image_idx;
    j["done_tagging"] = _done_tagging;
    j["image_paths"] =  _image_paths;
    return j;
}


std::unique_ptr<ManuallyTagger> ManuallyTagger::from_json(const nlohmann::json &json) {
    auto tagger = std::make_unique<ManuallyTagger>();
    tagger->_image_idx = json["image_idx"];
    std::vector<bool> done_tagging;
    for(auto done : json["done_tagging"]) {
        done_tagging.push_back(done);
    }
    tagger->_done_tagging = done_tagging;
    std::vector<std::string> image_paths;
    for(auto path : json["image_paths"]) {
        image_paths.push_back(path);
    }
    tagger->_image_paths  = image_paths;
    return tagger;
}
}
