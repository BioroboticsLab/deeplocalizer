

#include "Image.h"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using namespace deeplocalizer;
using boost::optional;

namespace io = boost::filesystem;

TEST_CASE( "ImageDesc", "[ImageDesc]" ) {
    std::vector<Tag> tag_vec{
            Tag(cv::Rect{}),
            Tag(cv::Rect{})
    };
    auto uniquePath = io::unique_path("/tmp/%%%%%%%%%%%.binary");
    SECTION("creation") {
        GIVEN("a filename") {
            std::string a_filename{"some_image.png"};
            THEN("a image description can be constructed") {
                ImageDesc descr{a_filename};
                REQUIRE(descr.filename == a_filename);
            }
        }
        GIVEN("a filename and a tag vector") {
            std::string a_filename{"some_image.png"};
            THEN("a image description can be constructed") {
                ImageDesc descr{a_filename, tag_vec};
                REQUIRE(descr.filename == a_filename);
                REQUIRE(descr.getTags() == tag_vec);
            }
        }
    }
    SECTION( "Serialization" ) {
        GIVEN( "an image description" ) {
            THEN("it can be saved and loaded") {
                ImageDesc desc{uniquePath.string(), tag_vec};
                desc.save();
                ImageDescPtr loaded_desc = ImageDesc::load(desc.savePath());
                REQUIRE(desc.filename== loaded_desc->filename);
            }
        }
        io::remove(uniquePath);
    }
}
