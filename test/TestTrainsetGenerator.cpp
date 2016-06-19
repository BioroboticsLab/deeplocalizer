
#define CATCH_CONFIG_RUNNER

#include <boost/filesystem.hpp>
#include <QtGui/QGuiApplication>
#include <QtGui/QPainter>
#include <QtCore/QTime>

#include "ManuallyTagger.h"
#include <catch.hpp>
#include "utils.h"
#include "qt_helper.h"
#include "TrainsetGenerator.h"

using namespace deeplocalizer;
using boost::optional;
using boost::none;
namespace io = boost::filesystem;

void printHistogram(const std::vector<TrainDatum> & data) {
    static const size_t nbins = 100;
    std::vector<size_t> bins(nbins, 0);
    for(const auto & datum : data) {
        size_t index = static_cast<size_t>(datum.taginess() * 100);
        ++bins[index];
    }
    for(size_t i = 0; i < bins.size(); i++) {
        const double fraction =  static_cast<double>(bins.at(i)) / data.size();
        std::string stars(static_cast<size_t>(1000*fraction), '*');
        std::cout << i << ": " << fraction << ", " << stars << std::endl;
    }
}

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
    SECTION("draw sample image") {
        for(size_t i = 0; i < 4; i++) {
            TrainsetGenerator gen;
            std::vector<TrainDatum> data;
            auto tags = cam2_desc.getTags();
            std::cout << "#TAGS: " << tags.size() << std::endl;
            QTime time;
            time.start();
            gen.process(cam2_desc, data);
            std::cout << "Time to generate data images: " << time.restart() <<  "ms" << std::endl;
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
            for(const auto & tag : cam2_desc.getTags()) {
            //    tag.draw(painter);
            }
            painter.end();
            std::stringstream ss;
            ss << "test_trainset_generator_" << i << ".png";
            std::cout << "save to: " << ss.str();
            qimage.save(QString::fromStdString(ss.str()));
            printHistogram(data);
        }
    }
}

int main( int argc, char** const argv )
{
    QGuiApplication qapp(argc, argv);
    int result = Catch::Session().run(argc, argv);
    return result;
}
