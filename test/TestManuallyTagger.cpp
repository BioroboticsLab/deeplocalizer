
#define CATCH_CONFIG_RUNNER

#include <QCoreApplication>
#include <boost/format.hpp>
#include <boost/filesystem/operations.hpp>
#include <chrono>

#include "catch.hpp"
#include "ManuallyTagger.h"

using namespace deeplocalizer;
namespace io = boost::filesystem;

class BigRandomManuallyTagger {
public:
    std::shared_ptr<ManuallyTagger> generate(int n_images) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> coordinate(0, 3000);
        std::uniform_int_distribution<int> n_tags(100, 150);

        std::vector<ImageDescPtr> descs;
        for(int i=0; i<n_images;i++) {
            std::vector<Tag> tags;
            for(int j=0; j<n_tags(gen);j++) {
                tags.emplace_back(Tag(cv::Rect(coordinate(gen),coordinate(gen), TAG_WIDTH, TAG_HEIGHT)));
            }
            io::path unique_path = io::unique_path("/tmp/test_tagger/%%%%%.jpeg");
            std::string filename = unique_path.string();
            io::create_directories(unique_path);
            {
                std::ofstream o(unique_path.string());
                o << "" << std::endl;
            }
            descs.push_back(std::shared_ptr<ImageDesc>(new ImageDesc(filename, tags)));
        }
        return std::make_shared<ManuallyTagger>(std::move(descs));
    }
    ~BigRandomManuallyTagger() {
        io::remove_all("/tmp/test_tagger/");
    }
};
TEST_CASE( "ManuallyTagger ", "[ManuallyTagger]" ) {
    std::vector<ImageDesc> image_descrs{
        ImageDesc{
            "testdata/with_5_tags.jpeg",
            std::vector<Tag>{Tag(cv::Rect{})}
        }
    };

    SECTION( " it can be constructed with a list of image descriptions" ) {
        GIVEN( "not existing images" ) {
            THEN(" it will throw an exception") {
                std::vector<ImageDesc>wrong_paths{
                    ImageDesc("noexistend.png")
                };
                REQUIRE_THROWS(new ManuallyTagger(wrong_paths));
            }
        }
        GIVEN( " some paths to existend images" ) {
            THEN(" it will construct an instance with valid default members") {
                ManuallyTagger tagger(image_descrs);
                auto pimgs = tagger.getImageDescs();

                REQUIRE(pimgs.size() == image_descrs.size());
                REQUIRE(*pimgs.at(0) == image_descrs.at(0));
            }
        }
    }

    SECTION( "sends signals" ) {
        ManuallyTagger * tagger = new ManuallyTagger(image_descrs);
        GIVEN("an index out of range") {
            THEN("it will emit the outOfRange signal") {
                bool outOfRangeEmitted = false;
                auto c = tagger->connect(tagger, &ManuallyTagger::outOfRange, [&](unsigned long idx){
                    REQUIRE(idx == ULONG_MAX);
                    outOfRangeEmitted = true;
                });
                tagger->loadImage(ULONG_MAX);
                REQUIRE(outOfRangeEmitted);
                tagger->disconnect(c);
            }
        }
        GIVEN("the first image") {
             THEN("it will emit the firstImage signal") {
                 bool firstImageEmitted = false;
                 auto c = tagger->connect(tagger, &ManuallyTagger::firstImage, [&]() {
                     firstImageEmitted = true;
                 });
                 tagger->loadImage(0);
                 REQUIRE(firstImageEmitted);
                 tagger->disconnect(c);
             }
        }
        GIVEN("the last image") {
            THEN("it will emit the lastImage signal") {
                bool lastImageEmitted = false;
                auto c = tagger->connect(tagger, &ManuallyTagger::lastImage, [&]() {
                    lastImageEmitted = true;
                });
                tagger->loadImage(0);
                REQUIRE(lastImageEmitted);
                tagger->disconnect(c);
            }
        }

        GIVEN("a call to the loadImage slot") {
            THEN("it will emit an loadedImage signal as soon the image is loaded") {
                bool loadedImageEmitted = false;
                auto c = tagger->connect(tagger, &ManuallyTagger::loadedImage,
                                         [&](unsigned long idx, ImageDescPtr  desc, ImagePtr  img) {
                    REQUIRE(idx == 0);
                    REQUIRE(*desc == *tagger->getImageDescs().at(0));
                    REQUIRE(*img == Image(*desc));
                    loadedImageEmitted = true;
                });
                tagger->loadImage(0);
                REQUIRE(loadedImageEmitted);
                tagger->disconnect(c);
            }
        }
    }
    SECTION("serialization") {
        GIVEN("many image descriptions") {
            using namespace std::chrono;
            unsigned int n_images = 10000;
            BigRandomManuallyTagger generator;
            auto tagger = generator.generate(n_images);
            time_point<system_clock> start_time = system_clock::now();
            THEN("it can be efficently be saved/loaded") {
                int times = 3;
                for(int i=0; i < times; i++) {
                    auto uniquePath = io::unique_path("/tmp/%%%%%%%%%%%.binary");
                    tagger->save(uniquePath.string());
                    io::remove(uniquePath);
                }
                auto elapsed = system_clock::now() - start_time;
                std::cout << "average save took: " << duration_cast<nanoseconds>(elapsed).count() / times
                    << "ns" << std::endl;
            }
        }
    }
    io::remove("testdata/with_5_tags.jpeg.proposal.desc");
}

int main( int argc, char** const argv )
{
    QCoreApplication * qapp = new QCoreApplication(argc, argv);
    int result = Catch::Session().run(argc, argv);
    return result;
}
