
#include <QCoreApplication>
#include <boost/program_options.hpp>
#include <chrono>
#include <functional>
#include <thread>

#include <ManuallyTagger.h>
#include <TrainsetGenerator.h>

#include "ProposalGenerator.h"
#include "qt_helper.h"
#include "utils.h"

using namespace deeplocalizer;
using namespace std::chrono;
using boost::optional;
using boost::make_optional;
namespace po = boost::program_options;
namespace io = boost::filesystem;

po::options_description desc_option("Options");
po::positional_options_description positional_opt;

time_point<system_clock> start_time;

void setupOptions() {
    desc_option.add_options()
            ("help,h", "Print help messages")
            ("test", po::value<std::string>(), "File with test image paths")
            ("train", po::value<std::string>(), "File with train image paths")
            ("format,f", po::value<std::string>(), "Format either `lmdb`, `images` or `all`. Default is `lmdb`. "
                                                   "`all` will save it both with lmdb and as images.")
            ("samples-per-tag,s", po::value<unsigned int>(), "Number of rotated and translated images per tag. Must be a multiple of 4."
                    " Default is 32.")
            ("output-dir,o", po::value<std::string>(), "Output images to this directory");
}


int run(QCoreApplication &,
        std::string trainfile,
        std::string testfile,
        Dataset::Format save_format,
        std::string output_dir,
        unsigned int samples_per_tag
) {
    std::cout << "loading training" << std::endl;
    const auto train_img_descs = ImageDesc::fromPathFile(trainfile, ManuallyTagger::IMAGE_DESC_EXT);
    const auto test_img_descs = ImageDesc::fromPathFile(testfile, ManuallyTagger::IMAGE_DESC_EXT);
    TrainsetGenerator gen{
            DataWriter::fromSaveFormat(output_dir, save_format)
    };
    gen.samples_per_tag = samples_per_tag;
    gen.wrong_samples_per_tag = samples_per_tag;
    std::cout << "Generating train set: " << std::endl;
    gen.processParallel(train_img_descs, Dataset::Train);
    std::cout << std::endl << std::endl;
    std::cout << "Generating test set: " << std::endl;
    gen.processParallel(test_img_descs, Dataset::Test);
    return 0;
}

void printUsage() {
    std::cout << "Usage: generate_proposals --test <TEST_FILE> --train <TRAIN_FILE> -f lmdb -o <DATA_DIR>"<< std::endl;
    std::cout << "    where <TEST_FILE> contains paths to images from which the test set will be generated."<< std::endl;
    std::cout << "          <TRAIN_FILE> contains paths to images from which the train set will be generated."<< std::endl;
    std::cout << "          <DATA_DIR> is the output directory."<< std::endl;
    std::cout << desc_option << std::endl;
}

int main(int argc, char* argv[])
{
    QCoreApplication qapp(argc, argv);
    deeplocalizer::registerQMetaTypes();
    setupOptions();
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc_option)
                      .positional(positional_opt).run(), vm);
    po::notify(vm);
    if (vm.count("help")) {
        printUsage();
        return 0;
    }
    if(vm.count("test") && vm.count("train") && vm.count("output-dir") && vm.count("format")) {
        auto format_str = vm.at("format").as<std::string>();
        auto opt_format = Dataset::parseFormat(format_str);
        if(not opt_format){
            std::cout << "No a valid Save format: " << format_str << std::endl;
            std::cout << "Save format must be either `lmdb`, `images` or `all`" << std::endl;
            return 1;
        }
        unsigned int samples_per_tag = 32;
        if(vm.count("samples-per-tag")) {
            samples_per_tag = vm.at("samples-per-tag").as<unsigned int>();
        }
        auto trainfile = vm.at("train").as<std::string>();
        auto testfile = vm.at("test").as<std::string>();
        auto output_dir = vm.at("output-dir").as<std::string>();
        return run(qapp, trainfile, testfile, opt_format.get(), output_dir, samples_per_tag);
    } else {
        std::cout << "No pathfile, format or output directory given." << std::endl;
        printUsage();
        return 1;
    }
}
