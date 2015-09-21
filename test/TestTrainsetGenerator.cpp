
#define CATCH_CONFIG_MAIN

#include <boost/filesystem.hpp>
#include <QPainter>

#include "ManuallyTagger.h"
#include "catch.hpp"
#include "utils.h"
#include "qt_helper.h"
#include "TrainsetGenerator.h"

using namespace deeplocalizer;
using boost::optional;
using boost::none;
namespace io = boost::filesystem;

TEST_CASE( "TrainsetGenerator", "" ) {
    Tag tag{cv::Rect{20, 20, TAG_WIDTH, TAG_HEIGHT}, none};
    if(not io::exists("tagger_images.txt")) {
        io::current_path(io::current_path() / "testdata");
    }
    ImageDesc img_desc{"one_tag_at_center.jpeg"};
    std::vector<ImageDesc> cam2_descs =
            ImageDesc::fromPathFile("tagger_images.txt", ManuallyTagger::IMAGE_DESC_EXT);
    ImageDesc cam2_desc = cam2_descs.front();
    std::vector<ImageDesc> img_descs{img_desc};
    Image img{img_desc};
    SECTION("trueSamples") {
        TrainsetGenerator gen;
        std::vector<TrainDatum> data;
        auto tags = cam2_desc.getTags();
        long n_yes = std::count_if(tags.cbegin(), tags.cend(),
                                   std::mem_fn(&Tag::isTag));
        GIVEN("a ImageDesc") {
            THEN("it will transform yes tags and randomly generate no tags") {
                gen.trueSamples(cam2_desc, data);
                CHECK(not data.empty());
                size_t n_true_samples = data.size();
                CHECK(n_true_samples == n_yes * gen.samples_per_tag);
                CHECK(std::all_of(data.begin(), data.end(), [](const auto & d) {
                    return d.tag().type() == TagType::IsTag;
                }));
                gen.wrongSamples(cam2_desc, data);
                CHECK(not data.empty());
                double n_wrong_samples = data.size() - n_true_samples;
                double expected_wrong_samples = n_yes * gen.samples_per_tag / TrainsetGenerator::RATIO_TRUE_TO_FALSE_SAMPLES;
                CHECK(n_wrong_samples / expected_wrong_samples  == Approx(1.).epsilon(0.01));
            }
        }
    }
    SECTION("draw sample image") {
        for(size_t i = 0; i < 4; i++) {
            TrainsetGenerator gen;
            std::vector<TrainDatum> data;
            auto tags = cam2_desc.getTags();
            gen.trueSamples(cam2_desc, data);
            gen.wrongSamples(cam2_desc, data);
            Image cam2_img(cam2_desc);
            cv::Mat cam2_color_img;
            // QPainter requires colored images
            cv::cvtColor(cam2_img.getCvMat(), cam2_color_img, CV_GRAY2BGR);
            QImage qimage = cvMatToQImage(cam2_color_img);
            QPainter painter(&qimage);
            ASSERT(painter.isActive(), "Expected painter to be active");
            for(size_t i = 0; i < data.size(); i++) {
                const auto & d = data.at(i);
                // pick only every 8th of the tags. Otherwise it would be to dense.
                if(i % 8 == 0) {
                    d.draw(painter);
                }
            }
            painter.end();
            std::stringstream ss;
            ss << "test_trainset_generator_" << i << ".jpeg";
            qimage.save(QString::fromStdString(ss.str()));
        }
    }
}
