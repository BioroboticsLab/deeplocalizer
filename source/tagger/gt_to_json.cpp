#include <boost/program_options.hpp>

#include <chrono>
#include "Image.h"
#include "utils.h"

using namespace deeplocalizer;
using namespace std::chrono;
namespace po = boost::program_options;
namespace io = boost::filesystem;
using boost::optional;

po::options_description desc_option("Options");
po::positional_options_description positional_opt;

time_point<system_clock> start_time;

void setupOptions() {
    desc_option.add_options()
            ("help,h", "Print help messages")
            ("pathfile", po::value<std::vector<std::string>>(), "File with paths");
    positional_opt.add("pathfile", 1);
}

std::vector<std::string> readPaths(const std::string &  pathfile) {
    std::vector<std::string> paths;
    std::string path_to_image;
    std::ifstream ifs{pathfile};
    for(int i = 0; std::getline(ifs, path_to_image); i++) {
        ASSERT(io::exists(path_to_image), "File " << path_to_image << " does not exists.");
        paths.push_back(path_to_image);
    }
    return paths;
}

std::string json_fname(const std::string & gt_fname) {
    io::path gt_path = gt_fname;
    gt_path.replace_extension("json");
    return gt_path.string();
}

int run(const std::string &  pathfile) {
    const auto paths = readPaths(pathfile);
    for(const auto & gt_fname : paths) {
        auto desc = ImageDesc::load(gt_fname);
        auto json_desc = desc->to_json();
        std::ofstream ofs(json_fname(gt_fname));
        ofs << json_desc.dump(4);
    }
    return 0;
}

int main(int argc, char* argv[])
{
    setupOptions();
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc_option)
                      .positional(positional_opt).run(), vm);
    po::notify(vm);
    if (vm.count("help")) {
        std::cout << "Usage: gt_to_json pathfile.txt "<< std::endl;
        std::cout << "    where pathfile.txt contains paths to gt boost binary files."<< std::endl;
        std::cout << desc_option << std::endl;
        return 0;
    }
    if(vm.count("pathfile")) {
        std::string pathfile = vm.at("pathfile").as<std::vector<std::string>>().at(0);
        run(pathfile);
    } else {
        std::cout << "No pathfile or output_dir are given" << std::endl;
        std::cout << "Usage: add_border [options] pathfile.txt "<< std::endl;
        std::cout << "    where pathfile.txt contains paths to images."<< std::endl;
        std::cout << desc_option << std::endl;
        return 0;
    }
    return 0;
}
