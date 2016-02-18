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
            ("output-dir,o",    po::value<std::string>(), "Write images to this directory")
            ("output-pathfile", po::value<std::string>()->default_value("images.txt"),
                 "Write output_pathfile to this directory. Default is <output_dir>/images.txt")
            ("pathfile",        po::value<std::vector<std::string>>(), "File with paths")
            ("border",          po::value<bool>()->default_value(true), "Add a border around the image.")
            ("use-hist-eq",     po::value<bool>()->default_value(false), "Apply local histogram equalization (CLAHE) to samples")
            ("use-threshold",   po::value<bool>()->default_value(false), "Apply adaptive thresholding to samples")
            ("binary-image",    po::value<bool>()->default_value(false), "Save binary image from thresholding")
            ("format,f",        po::value<std::string>()->default_value("jpeg"), "image output format. `png` or `jpeg`")
            ("compression,c",   po::value<int>(), "compression ratio")
            ("benchmark",       po::value<bool>()->default_value(false), "Try out different compression ratios and formats");
    positional_opt.add("pathfile", 1);
}

enum ImageFormat {
    PNG,
    JPEG

};

std::string format_to_str(ImageFormat format) {
    if (format == ImageFormat::JPEG) {
        return "jpeg";
    } else if(format == ImageFormat::PNG) {
        return "png";
    } else {
        return "wrong";
    }
}

struct PreprocessOptions {
    io::path output_dir;
    bool use_hist_eq;
    bool use_thresholding;
    bool use_binary_image;
    bool add_border;
    ImageFormat format;
    int compression;
    bool benchmark;
    std::pair<int, int> opencv_compression()const {
        int f;
        if (format == ImageFormat::JPEG) {
            f = CV_IMWRITE_JPEG_QUALITY;
        } else {
            f = CV_IMWRITE_PNG_COMPRESSION;
        }
        return std::make_pair(f, compression);
    };
};

// opencv default values
static const int DEFAULT_JPEG_COMPRESSION = 95;
static const int DEFAULT_PNG_COMPRESSION = 3;

io::path addWb(io::path filename, ImageFormat format) {
    io::path output_path(filename);
    output_path.replace_extension();
    output_path += std::string("_wb.") + format_to_str(format);
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
    std::cout << "Add border to " << nb_images << " images. Saved new images paths to: " << pathfile.string() << std::endl;
    std::cout << std::endl;
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
        auto output = addWb(opt.output_dir / input_path.filename(), opt.format);
        if(not img.write(output, opt.opencv_compression())) {
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

double preprocess(const std::vector<ImageDesc> image_descs,
        const io::path &  output_pathfile,
        const PreprocessOptions  & opt) {
    auto start = std::chrono::system_clock::now();
    io::create_directories(opt.output_dir);
    start_time = system_clock::now();
    printProgress(start_time, 0);
    const size_t nb_cpus = std::max(2*std::thread::hardware_concurrency(), 1u);
    std::vector<std::thread> threads;
    size_t nb_done = 0;

    pathss_t output_pathss;
    size_t part = image_descs.size()/nb_cpus;
    for(size_t i = 0; i < nb_cpus; i++) {
        std::shared_ptr<std::vector<std::string>> out_paths = std::make_shared<std::vector<std::string>>();
        out_paths->reserve(part);
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
    writeOutputPathfile(output_pathfile, output_pathss);
    std::chrono::duration<double> duration = std::chrono::system_clock::now() - start;
    return duration.count();
}

std::vector<std::pair<ImageFormat, int>> benchmark_formats() {
    std::vector<std::pair<ImageFormat, int>> formats;
    for(size_t c : std::vector<size_t>{75, 80, 85, 90}) {
        formats.emplace_back(std::make_pair(ImageFormat::JPEG, c));
    }
    for(size_t c : std::vector<size_t>{9, 6, 3, 0}) {
        formats.emplace_back(std::make_pair(ImageFormat::PNG, c));
    }
    return formats;
}

std::string mean_file_size(io::path dir) {
    io::directory_iterator end_itr;
    size_t nb_files = 0;
    size_t total_size = 0;

    for (io::directory_iterator itr(dir); itr != end_itr; ++itr)
    {
        const io::path & file = itr->path();
        if (io::is_regular_file(file)) {
            total_size += io::file_size(file);
            nb_files++;
        }
    }
    double mean_size = total_size / static_cast<double>(nb_files);
    const static size_t one_mb = 1 << 20;
    double size_in_mb = mean_size / one_mb;
    std::stringstream ss;
    ss << size_in_mb << "MB";
    return ss.str();
}

void benchmark(const std::vector<ImageDesc> image_descs,
        const io::path &  output_pathfile,
        const PreprocessOptions  & opt) {
    auto formats = benchmark_formats();
    std::vector<std::pair<PreprocessOptions, double>> results;
    for(auto & format_compression : formats) {
        PreprocessOptions bench_opt = opt;
        bench_opt.format = std::get<ImageFormat>(format_compression);
        bench_opt.compression = std::get<int>(format_compression);
        std::stringstream ss;
        ss << format_to_str(bench_opt.format) << "_c_" << bench_opt.compression;
        bench_opt.output_dir = opt.output_dir / ss.str();
        std::cout << "Benchmark: " << format_to_str(bench_opt.format) << ", Compression: " << bench_opt.compression << std::endl;
        double duration = preprocess(image_descs, output_pathfile, bench_opt);
        std::cout << "Done in: " << duration << "s" << std::endl;
        results.push_back(std::make_pair(bench_opt, duration));
    }
    for(size_t i = 0; i < formats.size(); i++) {
        PreprocessOptions bench_opt = std::get<PreprocessOptions>(results.at(i));
        std::cout << "Format: " << format_to_str(bench_opt.format) << ", Compression: " << bench_opt.compression << std::endl;
        std::cout << "file size: " << mean_file_size(bench_opt.output_dir) << std::endl;
        std::cout << "duration: " << std::get<double>(results.at(i)) << "s" << std::endl;
        std::cout << std::endl;
    }
}

int run(const std::vector<ImageDesc> image_descs,
        const io::path &  output_pathfile,
        const PreprocessOptions  & opt
        ) {
    if (opt.benchmark) {
        benchmark(image_descs, output_pathfile, opt);
    } else {
        double duration = preprocess(image_descs, output_pathfile, opt);
        std::cout << "Done in: " << duration << "s" << std::endl;
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

        io::path output_pathfile = vm.at("output-pathfile").as<std::string>();
        if (output_pathfile.is_relative()) {
            output_pathfile = output_dir / output_pathfile;
        }
        bool use_hist_eq = vm.at("use-hist-eq").as<bool>();
        bool use_threshold = vm.at("use-threshold").as<bool>();
        bool use_binary_image = vm.at("binary-image").as<bool>();
        bool benchmark = vm.at("benchmark").as<bool>();
        ImageFormat format;
        std::string str_format = vm.at("format").as<std::string>();
        if (str_format.compare("png") == 0) {
            format = ImageFormat::PNG;
        } else if (str_format.compare("jpeg") == 0) {
            format = ImageFormat::JPEG;
        } else {
            std::cout << "Expected `png` and `jpeg` format. But got: " << str_format << std::endl;
            exit(1);
        }

        int compression;
        if (vm.count("compression")) {
            compression = vm.at("compression").as<int>();
        } else if (format == ImageFormat::JPEG){
            compression = DEFAULT_JPEG_COMPRESSION;
        } else /* (format == ImageFormat::PNG) */ {
            compression = DEFAULT_PNG_COMPRESSION;
        }
        if (use_binary_image) {
            use_threshold = true;
        }
        bool add_border = vm.at("border").as<bool>();
        PreprocessOptions opt {
                output_dir,
                use_hist_eq,
                use_threshold,
                use_binary_image,
                add_border,
                format,
                compression,
                benchmark
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
