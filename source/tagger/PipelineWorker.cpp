#include "PipelineWorker.h"

#include <boost/filesystem.hpp>
#include <pipeline/datastructure/Tag.h>

#include "Tag.h"
#include "Image.h"
#include "qt_helper.h"
#include "utils.h"

namespace deeplocalizer {

using boost::optional;
namespace io = boost::filesystem;

const std::string PipelineWorker::DEFAULT_CONFIG_FILE = "pipeline-config.json";

const std::string PipelineWorker::DEFAULT_JSON_SETTINGS = R"({
    "PREPROCESSOR": {
        "COMB_DIFF_SIZE": "15",
        "COMB_ENABLED": "1",
        "COMB_LINE_COLOR": "0",
        "COMB_LINE_WIDTH": "9",
        "COMB_MAX_SIZE": "0",
        "COMB_MIN_SIZE": "0",
        "COMB_THRESHOLD": "255",
        "HONEY_AVERAGE_VALUE": "151",
        "HONEY_ENABLED": "1",
        "HONEY_FRAME_SIZE": "5",
        "HONEY_STD_DEV": "167",
        "OPT_AVERAGE_CONTRAST_VALUE": "0",
        "OPT_FRAME_SIZE": "500",
        "OPT_USE_CONTRAST_STRETCHING": "1",
        "OPT_USE_EQUALIZE_HISTOGRAM": "0"
    },
    "LOCALIZER": {
        "BINARY_THRESHOLD": "10",
        "EROSION_SIZE": "27",
        "FIRST_DILATION_NUM_ITERATIONS": "1",
        "FIRST_DILATION_SIZE": "10",
        "MAX_TAG_SIZE": "250",
        "MIN_BOUNDING_BOX_SIZE": "100",
        "SECOND_DILATION_SIZE": "3"
    },
    "ELLIPSEFITTER": {
        "CANNY_INITIAL_HIGH": "25",
        "CANNY_MEAN_MAX": "22",
        "CANNY_MEAN_MIN": "12",
        "CANNY_VALUES_DISTANCE": "23",
        "MAX_MAJOR_AXIS": "53",
        "MAX_MINOR_AXIS": "65",
        "MIN_MAJOR_AXIS": "26",
        "MIN_MINOR_AXIS": "25",
        "THRESHOLD_BEST_VOTE": "1740",
        "THRESHOLD_EDGE_PIXELS": "45",
        "THRESHOLD_VOTE": "1400",
        "USE_XIE_AS_FALLBACK": "1"
        }
})";

PipelineWorker::PipelineWorker() {
    init(DEFAULT_CONFIG_FILE);
}

PipelineWorker::PipelineWorker(const std::string & config_file) {
    init(config_file);
}

void PipelineWorker::init(const std::string & config_file) {
    using namespace pipeline;
    io::path config_path{config_file};
    boost::property_tree::ptree pt;
    if(io::exists(config_path)) {
        boost::property_tree::read_json(config_path.string(), pt);
    } else {
        std::stringstream ss;
        ss << PipelineWorker::DEFAULT_JSON_SETTINGS;
        boost::property_tree::read_json(ss, pt);
    }
    settings::preprocessor_settings_t pre_settings;
    settings::localizer_settings_t loc_settings;
    settings::ellipsefitter_settings_t ell_settings;

    pre_settings.loadValues(pt);
    loc_settings.loadValues(pt);
    ell_settings.loadValues(pt);

    _preprocessor.loadSettings(pre_settings);
    _localizer.loadSettings(loc_settings);
    _ellipseFitter.loadSettings(ell_settings);
}

std::vector<Tag> PipelineWorker::tagsProposals(ImageDesc & img_descr) {
    Image img{img_descr};
    cv::Mat preprocced = _preprocessor.process(img.getCvMat());
    std::vector<pipeline::Tag> localizer_tags =
            _localizer.process(std::move(img.getCvMat()), std::move(preprocced));
    const std::vector<pipeline::Tag> pipeline_tags =
            _ellipseFitter.process(std::move(localizer_tags));
    std::vector<Tag> tags;
    for(auto ptag : pipeline_tags) {
        tags.emplace_back(Tag(ptag));
    }
    return tags;
}

void PipelineWorker::process(ImageDesc img) {
    ASSERT(io::exists(img.filename),
           "Could not open file: `" << img.filename << "`");
    img.setTags(this->tagsProposals(img));
    emit resultReady(img);
}
void PipelineWorker::findEllipse(cv::Mat mat, Tag tag) {
    pipeline::Tag pipeTag(tag.getBoundingBox(), 0 /* id */);
    pipeTag.setOrigSubImage(mat);
    auto pipelineTags= _ellipseFitter.process({pipeTag});
    Tag tagWithEll(pipelineTags.at(0));
    tagWithEll.setId(tag.id());
    tagWithEll.setType(tag.type());
    emit tagWithEllipseReady(tagWithEll);
}
}
