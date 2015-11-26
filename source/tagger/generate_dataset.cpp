
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
#include "deeplocalizer_tagger.h"

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
            ("format,f", po::value<std::string>()->default_value("hdf5"), "Format either `images`, `hdf5`. Default is `hdf5`. ")
            ("sample-rate,s", po::value<double>()->default_value(32), "Number samples per tag.")
            ("scale,c", po::value<double>()->default_value(1), "Scale applied to the tag images. Default is 1.")
            ("use-rotation,r", po::value<bool>()->default_value(true), "Rotate subimages with a tagness over 0.8")
            ("pathfile", po::value<std::string>(), "Pathfile to the images")
            ("output-dir,o", po::value<std::string>(), "Output images to this directory");
    positional_opt.add("pathfile", 1);
}


int run(QCoreApplication &,
        std::string pathfile,
        Dataset::Format save_format,
        std::string output_dir,
        double sample_rate,
        double scale,
        bool use_rotation
) {
    std::cout << "loading training" << std::endl;
    const auto img_descs = ImageDesc::fromPathFile(pathfile, ManuallyTagger::IMAGE_DESC_EXT);
    TrainsetGenerator gen(
            DataWriter::fromSaveFormat(output_dir, save_format),
            sample_rate, scale, use_rotation);
    std::cout << "Generating data set: " << std::endl;
    gen.processParallel(img_descs);
    std::cout << "Saved dataset to: " << output_dir << std::endl;
    return 0;
}

void printUsage() {
    std::cout << "Usage: generate_proposals  -f hdf5 -o <DATA_DIR> <FILE>"<< std::endl;
    std::cout << "    where <FILE> contains paths to images"<< std::endl;
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
    if(vm.count("pathfile") && vm.count("output-dir")) {
        auto format_str = vm.at("format").as<std::string>();
        auto opt_format = Dataset::parseFormat(format_str);
        if(not opt_format){
            std::cout << "No a valid Save format: " << format_str << std::endl;
            std::cout << "Save format must be either `lmdb`, `images`, `hdf5` or `all`" << std::endl;
            return 1;
        }
        double sample_rate = vm.at("sample-rate").as<double>();
        auto pathfile = vm.at("pathfile").as<std::string>();
        auto output_dir = vm.at("output-dir").as<std::string>();
        double scale = vm.at("scale").as<double>();
        bool use_rotation = vm.at("use-rotation").as<bool>();
        try {
            return run(qapp, pathfile, opt_format.get(), output_dir, sample_rate,
                       scale, use_rotation);
        } catch(const std::exception & e) {
            std::cerr << "An Exception occurred: " << e.what() << std::endl;
            return 1;
        }
    } else {
        std::cout << "No pathfile, format or output directory given." << std::endl;
        printUsage();
        return 1;
    }
}
