#define CATCH_CONFIG_RUNNER

#include <QCoreApplication>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>

#include <catch.hpp>
#include <json.hpp>

#include "Image.h"
#include "utils.h"
#include "qt_helper.h"

namespace io = boost::filesystem;
using boost::optional;
using namespace deeplocalizer;
using json = nlohmann::json;

TEST_CASE( "JSON serialization", "[serialize]" ) {
    ImageDesc img("image_path.jpeg");
    Tag tag(cv::Rect(30, 40, TAG_WIDTH, TAG_HEIGHT));
    Tag tag_with_ell(cv::Rect(300, 400, TAG_WIDTH, TAG_HEIGHT));
    SECTION("simple tag") {
        json j = tag.to_json();
        Tag from_json = Tag::from_json(j);
        REQUIRE(tag == from_json);
    }
    SECTION("tag with ellipse") {
        json j = tag_with_ell.to_json();
        Tag from_json = Tag::from_json(j);
        REQUIRE(tag_with_ell == from_json);
    }
    img.addTag(tag);
    img.addTag(tag_with_ell);
    SECTION("image desc") {
        json j = img.to_json();
        ImageDesc from_json = ImageDesc::from_json(j);
        REQUIRE(from_json == img);
    }
}

int main( int argc, char** const argv )
{
    QCoreApplication * qapp = new QCoreApplication(argc, argv);
    registerQMetaTypes();
    int exit_code = Catch::Session().run(argc, argv);
    return exit_code;
}
