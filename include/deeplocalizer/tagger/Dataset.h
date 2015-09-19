
#ifndef DEEP_LOCALIZER_DATASET_H
#define DEEP_LOCALIZER_DATASET_H


#include <memory>
#include <boost/filesystem.hpp>
#include <mutex>
#include <lmdb.h>

#include "TrainDatum.h"

namespace deeplocalizer {

namespace Dataset {
    enum Phase {
        Train,
        Test
    };
    enum Format {
        Images,
        LMDB,
        All,
        HDF5,
        DevNull
    };
    const double TEST_PARTITION = 0.15;
    const double TRAIN_PARTITION = 1 - TEST_PARTITION;
    boost::optional<Format> parseFormat(const std::string &str);
}

}
#endif //DEEP_LOCALIZER_DATASET_H
