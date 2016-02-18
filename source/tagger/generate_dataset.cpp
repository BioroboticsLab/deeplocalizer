
#include <QCoreApplication>
#include <boost/program_options.hpp>
#include <chrono>
#include <functional>
#include <thread>

#include <ManuallyTagger.h>
#include <TrainsetGenerator.h>
#include <QPainter>
#include <QTime>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGuiApplication>
#include <QGraphicsPixmapItem>

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
            ("sample-rate,s", po::value<double>()->default_value(32),
             "The sample-rate determines the number of tags to sample. A sample rate of 32 means that for every actual tag, 32 true and false samples generated.")
            ("scale,c", po::value<double>()->default_value(1), "Scale applied to the tag images. Default is 1.")
            ("use-rotation,r", po::value<bool>()->default_value(true), "Rotate subimages with a tagness over 0.8")
            ("acceptance-rate,a", po::value<double>()->default_value(0.05),
             "The rate at which false images are accepted. 0 means never and 1. means allways. "
             "A value around 0.05-0.15 should work fine.")
            ("pathfile", po::value<std::string>(), "Pathfile to the images")
            ("show", po::value<bool>()->default_value(false), "Visualise the sampling for the first image in the pathfile.")
            ("output-dir,o", po::value<std::string>(), "Output images to this directory");
    positional_opt.add("pathfile", 1);
}


int run(const std::vector<ImageDesc> & img_descs,
        Dataset::Format save_format,
        std::string output_dir,
        double sample_rate,
        double scale,
        bool use_rotation,
        double acceptance_rate
) {
    std::cout << "loading training" << std::endl;
    TrainsetGenerator gen(
            DataWriter::fromSaveFormat(output_dir, save_format),
            sample_rate, scale, use_rotation, acceptance_rate);
    std::cout << "Generating data set: " << std::endl;
    gen.processParallel(img_descs);
    std::cout << "Saved dataset to: " << output_dir << std::endl;
    return 0;
}

int run_show(QGuiApplication & qapp,
         const std::vector<ImageDesc> & img_descs,
         double sample_rate,
         double scale,
         bool use_rotation,
         double acceptance_rate) {
    std::vector<TrainDatum> data;
    TrainsetGenerator gen(std::make_unique<DevNullWriter>(),
                          sample_rate, scale, use_rotation, acceptance_rate);

    auto desc = img_descs.front();
    auto tags = desc.getTags();
    std::cout << "#TAGS: " << tags.size() << std::endl;
    QTime time;
    time.start();
    gen.process(desc, data);
    std::cout << "Time to generate data images: " << time.restart() <<  "ms" << std::endl;
    Image img(desc);
    cv::Mat color_img;
    // QPainter requires colored images
    cv::cvtColor(img.getCvMat(), color_img, CV_GRAY2BGR);
    QImage qimage = cvMatToQImage(color_img);
    QPainter painter(&qimage);
    ASSERT(painter.isActive(), "Expected painter to be active");
    for(const auto & d : data) {
        d.draw(painter);
    }
    painter.end();
    qimage.save("show_sampling.png");

    std::cout << "Saved image to show_sampling.png" << std::endl;
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
        double acceptance_rate = vm.at("acceptance-rate").as<double>();
        bool show = vm.at("show").as<bool>();
        try {
            const auto img_descs = ImageDesc::fromPathFile(pathfile, ManuallyTagger::IMAGE_DESC_EXT);
            if (show) {
                QGuiApplication qapp(argc, argv);
                run_show(qapp, img_descs, sample_rate, scale, use_rotation, acceptance_rate);

            } else {
                return run(img_descs, opt_format.get(), output_dir, sample_rate,
                           scale, use_rotation, acceptance_rate);
            }
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
