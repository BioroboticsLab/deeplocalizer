

#define CATCH_CONFIG_RUNNER

#include <QCoreApplication>
#include <QTimer>
#include <boost/format.hpp>
#include <boost/filesystem/operations.hpp>

#include "catch.hpp"
#include "ManuallyTagger.h"
#include "ProposalGenerator.h"
#include "qt_helper.h"

using namespace deeplocalizer;
namespace io = boost::filesystem;
using boost::optional;

static int exit_code = 1;
static int argc= 1;
static char ** argv;

static ProposalGenerator * gen;

void registerQuit(ProposalGenerator * gen) {
    gen->connect(gen,
                 &ProposalGenerator::finished, QCoreApplication::instance(),
                 [&]() {
                      QCoreApplication::instance()->exit(exit_code);
                 }, Qt::QueuedConnection);
}

TEST_CASE( "ProposalGenerator", "[ProposalGenerator]" ) {
    QCoreApplication * qapp = new QCoreApplication(argc, argv);
    std::string img_path{"testdata/with_5_tags.jpeg"};
    io::path unique_img = io::temp_directory_path() / io::unique_path("%%%%%%%%%.jpeg");
    io::copy(img_path, unique_img);
    std::vector<std::string> image_paths = {unique_img.string()};
    gen = new ProposalGenerator(image_paths);
    deeplocalizer::registerQMetaTypes();
    QTimer * timer = new QTimer(qapp);
    qapp->connect(timer, &QTimer::timeout, qapp,
            std::bind(&QCoreApplication::exit, exit_code),
            Qt::QueuedConnection);
    // wait at maximum 2 minutes
    timer->start(2*60*1000);
    ImageDesc img("image_path.jpeg");
    Tag tag(cv::Rect(0, 0, 10, 20), optional<pipeline::Ellipse>());
    registerQuit(gen);
    SECTION( "runs images through the pipeline" ) {
        bool finishedEmitted = false;
        GIVEN("it is finished with processing an image with bees") {
            THEN ("it will give you tags proposals "
                          "and the todo list will be empty") {
                gen->connect(gen, &ProposalGenerator::finished, gen, [&]() {
                                 REQUIRE(not gen->getProposalImages().empty());
                                 auto &img = gen->getProposalImages().front();
                                 REQUIRE(img.getTags().size() > 0);
                                 finishedEmitted = true;
                }, Qt::QueuedConnection);
                gen->connect(gen, &ProposalGenerator::finished, gen, [&]() {
                    REQUIRE(gen->getBeforePipelineImages().empty());
                }, Qt::QueuedConnection);
            }
        }
        gen->processPipeline();
        qapp->exec();
        REQUIRE(finishedEmitted);
    }
    SECTION("serialization") {
        bool finishedEmitted = false;
        GIVEN("an image with tags") {
            THEN(" the tags can be saved and reloaded") {
                gen->connect(gen, &ProposalGenerator::finished, gen, [&]() {
                    auto load_imgs = ImageDesc::fromPaths(image_paths);
                    REQUIRE(gen->getProposalImages().size() ==
                            load_imgs.size());
                    for (unsigned int i = 1;
                         i < gen->getProposalImages().size(); i++) {
                        auto gen_img = gen->getProposalImages().at(i);
                        auto load_img = load_imgs.at(i);
                        REQUIRE(gen_img == load_img);
                    }
                    finishedEmitted = true;
                }, Qt::QueuedConnection);
                gen->processPipeline();
            }
        }
        gen->processPipeline();
        qapp->exec();
        REQUIRE(finishedEmitted);
    }
    io::remove(unique_img);
}

int main( int _argc, char** const _argv )
{
    argc = _argc;
    argv = _argv;
    exit_code = Catch::Session().run(argc, argv);
    return exit_code;
}
