
#define CATCH_CONFIG_MAIN

#include <boost/filesystem.hpp>
#include <QPainter>

#include "ManuallyTagger.h"
#include "catch.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <utils.h>
#include "deeplocalizer/classifier/DataReader.h"
#include <caffe/util/io.hpp>
#include "TrainsetGenerator.h"

using namespace deeplocalizer;
using boost::optional;
using boost::none;
namespace io = boost::filesystem;

TEST_CASE( "TestDataReaderWriter", "" ) {
    Tag tag{cv::Rect{20, 20, TAG_WIDTH, TAG_HEIGHT}, none};
    if(not io::exists("tagger_images.txt")) {
        io::current_path(io::current_path() / "testdata");
    }
    std::vector<ImageDesc> cam2_descs =
            ImageDesc::fromPathFile("tagger_images.txt", ManuallyTagger::IMAGE_DESC_EXT);
    ImageDesc cam2_desc = cam2_descs.front();
    io::path unique_path = io::unique_path("/tmp/test_deeplocalizer/%%%%%%/");
    io::create_directories(unique_path);
    std::vector<int> shape{32, 1, TAG_WIDTH, TAG_HEIGHT};
    SECTION("Reader / Writer") {
        GIVEN("a ImageDesc") {
            THEN("Writer will save the samples and Reader will load it.") {
                TrainsetGenerator gen;
                std::vector<TrainDatum> data;
                gen.process(cam2_desc, data);
                auto tags = cam2_desc.getTags();
                {
                    ImageWriter imageWriter((unique_path / "images").string());
                    REQUIRE(not data.empty());
                    imageWriter.write(data);
                }
                {
                    ImageReader imageReader((unique_path / "images" / "images.txt").string(), shape);
                    caffe::Blob<float> blob;
                    std::vector<int> labels;
                    REQUIRE(imageReader.read(blob, labels));
                    REQUIRE(std::equal(blob.shape().cbegin(), blob.shape().cend(), shape.cbegin()));
                }
            }
        }
        io::remove_all(unique_path);
    }
}

