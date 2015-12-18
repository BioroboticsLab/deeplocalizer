#include <boost/program_options.hpp>

#include <chrono>
#include <thread>
#include <mutex>
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
            ("output-dir,o", po::value<std::string>(), "Write images to this directory")
            ("output-pathfile", po::value<std::string>(),
             "Write output_pathfile to this directory. Default is <output_dir>/images.txt")
            ("pathfile", po::value<std::vector<std::string>>(), "File with paths")
            ("border", po::value<bool>()->default_value(true), "Add a border around the image.")
            ("use-hist-eq", po::value<bool>()->default_value(false), "Apply local histogram equalization (CLAHE) to samples")
            ("use-threshold", po::value<bool>()->default_value(false), "Apply adaptive thresholding to samples")
            ("binary-image", po::value<bool>()->default_value(false), "Save binary image from thresholding");
    positional_opt.add("pathfile", 1);
}
struct PreprocessOptions {
    io::path output_dir;
    bool use_hist_eq;
    bool use_thresholding;
    bool use_binary_image;
    bool add_border;
};

io::path addWb(io::path filename) {
    io::path output_path(filename);
    auto extension = output_path.extension();
    output_path.replace_extension();
    output_path += "_wb" + extension.string();
    return output_path;
}

using pathss_t = std::vector<std::shared_ptr<std::vector<std::string>>>;

void writeOutputPathfile(io::path pathfile, const pathss_t &output_pathss) {
    std::ofstream of(pathfile.string());
    size_t nb_images = 0;
    for (const auto &paths : output_pathss) {
        nb_images += paths->size();
        for (const auto & path : *paths) {
            of << path << '\n';
        }
    }
    of << std::flush;

    std::cout << std::endl;
    std::cout << "Add border to " << nb_images << " images. Saved new images paths to: " << std::endl;
    std::cout << pathfile.string() << std::endl;
}

void adaptiveTresholding(cv::Mat & mat, bool use_binary_image) {
    static double max_value = 255;
    static size_t block_size = 51;
    static double weight_original = 0.7;
    static double weight_threshold = 0.3;

    cv::Mat mat_threshold(mat.rows, mat.cols, CV_8UC1);
    cv::adaptiveThreshold(mat, mat_threshold, max_value,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY, block_size, 0);
    if (use_binary_image) {
        mat = mat_threshold;
    } else {
        cv::addWeighted(mat, weight_original, mat_threshold, weight_threshold, 0 /*gamma*/, mat);
    }
}

void localHistogramEq(cv::Mat & mat) {
    static const int clip_limit = 2;
    static const cv::Size tile_size(deeplocalizer::TAG_WIDTH, deeplocalizer::TAG_HEIGHT);
    auto clahe = cv::createCLAHE(clip_limit, tile_size);
    cv::Mat image_clahe;
    clahe->apply(mat, mat);
}

void makeBorder(cv::Mat & mat) {
    auto mat_with_border = cv::Mat(mat.rows + TAG_HEIGHT,
                                   mat.cols + TAG_WIDTH, CV_8U);
    cv::copyMakeBorder(mat, mat_with_border,
                       TAG_HEIGHT / 2, TAG_HEIGHT / 2,
                       TAG_WIDTH  / 2, TAG_WIDTH  / 2,
                       cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
    mat = mat_with_border;
}
void processImage(Image & img, const PreprocessOptions & opt) {
    cv::Mat & mat = img.getCvMatRef();

    if (opt.add_border) {
        makeBorder(mat);
    }
    if (opt.use_hist_eq) {
        localHistogramEq(mat);
    }
    if (opt.use_thresholding) {
        adaptiveTresholding(mat, opt.use_binary_image);
    }
}
void threadWorkerFn(const std::vector<ImageDesc> & image_descs,
                    const size_t start, const size_t end,
                    std::shared_ptr<std::vector<std::string>>  output_paths,
                    const PreprocessOptions & opt,
                    size_t & nb_done,
                    std::mutex & cout_mutex) {
    for(size_t i = start; i < end; i++) {
        const ImageDesc & desc = image_descs.at(i);
        Image img(desc);
        processImage(img, opt);
        auto input_path =  io::path(desc.filename);
        auto output = addWb(opt.output_dir / input_path.filename());
        if(not img.write(output)) {
            std::lock_guard<std::mutex> look(cout_mutex);
            std::cerr << "Fail to write image : " << output.string() << std::endl;
            return;
        }
        output_paths->push_back(output.string());
        {
            std::lock_guard<std::mutex> look(cout_mutex);
            nb_done++;
            printProgress(start_time, static_cast<double>(nb_done)/image_descs.size());
        }
    }
}
int run(const std::vector<ImageDesc> image_descs,
        optional<io::path> output_pathfile,
        const PreprocessOptions  & opt
        ) {
    io::create_directories(opt.output_dir);
    start_time = system_clock::now();
    printProgress(start_time, 0);
    const size_t nb_cpus = std::max(static_cast<unsigned int>(1.5*std::thread::hardware_concurrency()), 1u);
    std::vector<std::thread> threads;
    size_t nb_done = 0;

    pathss_t output_pathss;
    size_t part = image_descs.size()/nb_cpus;
    for(size_t i = 0; i < nb_cpus; i++) {
        std::shared_ptr<std::vector<std::string>> out_paths = std::make_shared<std::vector<std::string>>();
        out_paths->reserve(2*part);
        output_pathss.push_back(out_paths);
    }
    for(size_t i = 0; i < nb_cpus; i++) {
        size_t start = i*part;
        size_t end = (i+1)*part;
        if (i + 1 == nb_cpus)  {
            end = image_descs.size();
        }
        std::mutex cout_look;
        threads.push_back(std::thread(&threadWorkerFn, std::cref(image_descs),
                                      start, end,
                                      output_pathss.at(i),
                                      std::cref(opt), std::ref(nb_done), std::ref(cout_look)));
    }
    for(auto & thread : threads) {
        thread.join();
    }
    writeOutputPathfile(output_pathfile.get_value_or(opt.output_dir / "images.txt"), output_pathss);
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
        std::cout << "Usage: add_border [options] pathfile.txt "<< std::endl;
        std::cout << "    where pathfile.txt contains paths to images."<< std::endl;
        std::cout << desc_option << std::endl;
        return 0;
    }
    if(vm.count("pathfile") && vm.count("output-dir")) {
        std::string pathfile =
                vm.at("pathfile").as<std::vector<std::string>>().at(0);
        auto image_descs = ImageDesc::fromPathFile(pathfile);
        auto output_dir = io::path(vm.at("output-dir").as<std::string>());

        optional<io::path> output_pathfile;
        if(vm.count("output-pathfile")) {
            output_pathfile = boost::make_optional(
                    io::path(vm.at("output-pathfile").as<std::string>()));
        }
        bool use_hist_eq = vm.at("use-hist-eq").as<bool>();
        bool use_threshold = vm.at("use-threshold").as<bool>();
        bool use_binary_image = vm.at("binary-image").as<bool>();
        if (use_binary_image) {
            use_threshold = true;
        }
        bool add_border = vm.at("border").as<bool>();
        PreprocessOptions opt {
                output_dir,
                use_hist_eq,
                use_threshold,
                use_binary_image,
                add_border
        };
        run(image_descs, output_pathfile, opt);
    } else {
        std::cout << "No pathfile or output_dir are given" << std::endl;
        std::cout << "Usage: add_border [options] pathfile.txt "<< std::endl;
        std::cout << "    where pathfile.txt contains paths to images."<< std::endl;
        std::cout << desc_option << std::endl;
        return 0;
    }
    return 0;
}
