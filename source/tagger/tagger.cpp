
#include "ManuallyTaggerWindow.h"
#include "qt_helper.h"
#include <QApplication>
#include <boost/program_options.hpp>
#include <ProposalGenerator.h>

using namespace deeplocalizer;

namespace po = boost::program_options;
namespace io = boost::filesystem;

po::options_description desc_option("Options");
po::positional_options_description positional_opt;

void setupOptions() {
    desc_option.add_options()
            ("help,h", "Print help messages")
            ("pathfile", po::value<std::vector<std::string>>(), "File with the paths to the images");

    positional_opt.add("pathfile", 1);
}
int run(QApplication & qapp, std::string pathfile) {
    std::unique_ptr<ManuallyTaggerWindow> window;
    if (io::exists(ManuallyTagger::DEFAULT_SAVE_PATH)) {
        auto tagger = ManuallyTagger::load(ManuallyTagger::DEFAULT_SAVE_PATH);
        window = std::make_unique<ManuallyTaggerWindow>(std::move(tagger));
    } else {
        auto proposals = ImageDesc::fromPathFilePtr(pathfile,
                                                    ProposalGenerator::IMAGE_DESC_EXT);
        window = std::make_unique<ManuallyTaggerWindow>(std::move(proposals));
    }
    window->show();
    return qapp.exec();
}

void printUsage() {
    std::cout << "Usage: generate_proposals [options] pathfile.txt "<< std::endl;
    std::cout << "    where pathfile.txt contains paths to images."<< std::endl;
    std::cout << desc_option << std::endl;
}
int main(int argc, char* argv[])
{
    QApplication qapp(argc, argv);
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
    if(vm.count("pathfile")) {
        auto pathfile = vm.at("pathfile").as<std::vector<std::string>>().at(0);
        return run(qapp, pathfile);
    } else {
        std::cout << "No proposal file or output directory given." << std::endl;
        printUsage();
        return 1;
    }
}

