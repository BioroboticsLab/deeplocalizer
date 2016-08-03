#pragma  once

#include <iostream>
#include <fstream>
#include <chrono>
#include <sstream>
#include <random>

#include <boost/serialization/nvp.hpp>
#include <boost/archive/archive_exception.hpp>
#include <boost/filesystem.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <json.hpp>

#ifndef NDEBUG
#   define ASSERT(condition, message) \
    do { \
        if (! (condition)) { \
            std::stringstream ss; \
            ss << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message; \
            std::cerr << ss.str() << std::endl; \
            throw ss.str(); \
        } \
    } while (false)
#else
#   define ASSERT(condition, message) \
    do {  \
        if (! (condition)) { \
            std::stringstream ss; \
            ss << message; \
            std::cerr << ss.str() << std::endl; \
            throw ss.str(); \
        } \
    } while (false)
#endif

namespace deeplocalizer {

template<typename T>
void safe_serialization(const std::string &path, const T && json) {
    boost::filesystem::path save_path{path};
    boost::filesystem::path tmp_path = boost::filesystem::unique_path(save_path.parent_path() / "%%%%%%%%%.json");
    std::ofstream os(tmp_path.string());
    os << json.dump(2);
    boost::filesystem::rename(tmp_path, save_path);
}

template<typename Clock>
inline void printProgress(const std::chrono::time_point<Clock> & start_time,
                   double progress) {
    using namespace std::chrono;
    using std::cout;
    int width = 40;
    auto elapsed = Clock::now() - start_time;
    if (progress <= 1e-5 || duration_cast<milliseconds>(elapsed).count() <= 100) {
        return;
    }
    unsigned long progress_chars = std::lround(width * progress);
    auto crosses = std::string(progress_chars, '#');
    auto spaces = std::string(width - progress_chars, ' ');
    cout << "\r " << static_cast<int>(progress * 100) << "% ["
    << crosses << spaces << "] ";
    auto eta = elapsed / progress - elapsed;
    auto h = duration_cast<hours>(eta).count();
    auto m = duration_cast<minutes>(eta).count() - 60 * h;
    auto s = duration_cast<seconds>(eta).count() - 60 * m - 60*60*h;
    cout << "eta ";
    if (h) {
        cout << h << "h ";
    }
    if (h || m) {
        cout << m << "m ";
    }
    cout << s << "s";
    cout << "          " << std::flush;
}

inline std::vector<unsigned long> shuffledIndecies(unsigned long n) {
    std::vector<unsigned long> indecies;
    indecies.reserve(n);
    for(unsigned long i = 0; i < n; i++) {
        indecies.push_back(i);
    }
    std::shuffle(indecies.begin(), indecies.end(), std::default_random_engine());
    return indecies;
}

inline std::vector<std::string>  parsePathfile(std::string path) {
    const boost::filesystem::path pathfile(path);
    ASSERT(boost::filesystem::exists(pathfile), "File " << pathfile << " does not exists.");
    std::ifstream ifs{pathfile.string()};
    std::string path_to_image;
    std::vector<std::string> paths;
    for(int i = 0; std::getline(ifs, path_to_image); i++) {
        ASSERT(boost::filesystem::exists(path_to_image), "File " << path_to_image << " does not exists.");
        paths.push_back(path_to_image);
    }
    return paths;
}

}
