#pragma once

#include <memory>
#include <random>
#include <boost/filesystem.hpp>
#include <mutex>

#include "TrainDatum.h"

namespace deeplocalizer {

namespace Dataset {
    enum Phase {
        Train,
        Test
    };
    enum Format {
        Images,
        HDF5,
        DevNull
    };
    const double TEST_PARTITION = 0.15;
    const double TRAIN_PARTITION = 1 - TEST_PARTITION;
    boost::optional<Format> parseFormat(const std::string &str);
}

}
