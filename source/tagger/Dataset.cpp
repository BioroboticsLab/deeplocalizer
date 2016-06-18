#include "Dataset.h"

#include <iomanip>
#include "utils.h"

namespace deeplocalizer {


namespace io = boost::filesystem;

boost::optional<Dataset::Format> Dataset::parseFormat(const std::string &str) {
    if(str == "images") {
        return Dataset::Format::Images;
    } else if (str == "hdf5") {
        return Dataset::Format::HDF5;
    } else {
        return boost::optional<Format>();
    }
}
}
