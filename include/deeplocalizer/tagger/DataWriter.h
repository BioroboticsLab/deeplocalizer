#ifndef DEEP_LOCALIZER_DATAWRITER_H
#define DEEP_LOCALIZER_DATAWRITER_H


#include <memory>
#include <boost/filesystem.hpp>
#include <mutex>
#include <hdf5.h>
#include <hdf5_hl.h>

#include "TrainDatum.h"
#include "Dataset.h"

namespace deeplocalizer {

class DataWriter {
public:
    virtual void write(const std::vector<TrainDatum> &dataset) = 0;
    virtual ~DataWriter() = default;
    static std::unique_ptr<DataWriter> fromSaveFormat(
            const std::string &output_dir,
            Dataset::Format format);
};


class ImageWriter : public DataWriter {
public:
    ImageWriter(const std::string & output_dir);
    virtual void write(const std::vector<TrainDatum> &dataset);
    virtual ~ImageWriter() = default;
private:
    boost::filesystem::path _output_dir;

    std::ofstream _stream;
    std::mutex _mutex;

    void writeImages(const std::vector<TrainDatum> &data) const;
    void writeLabelFile(const std::vector<TrainDatum> &data);
    inline std::string filename(const TrainDatum & datum) const {
        return datum.description() + ".jpeg";
    }
    inline std::pair<std::string, double> toFilenameLabel(const TrainDatum & datum) {
        return std::make_pair(filename(datum), datum.taginess());
    }
};


class HDF5Writer : public DataWriter {
public:
    HDF5Writer(const std::string &output_dir);
    virtual void write(const std::vector<TrainDatum> &dataset);
    virtual ~HDF5Writer();

private:
    void saveLabelHDF5Dataset(const hid_t file_id);
    void saveDataHDF5Dataset(const hid_t file_id);
    void writeBufferToFile();
    std::string nextFilename();
    const size_t MAX_HDF5_FILE = 1 << 29;  // ~ 512MB
    std::ofstream _txt_file;

    boost::filesystem::path _output_dir;
    std::string _unique_path_format;
    std::vector<TrainDatum> _buffer;
    size_t _max_buffer_size = 0;
    std::mutex _mutex;
    unsigned long _file_id = 0;
    size_t _mat_size = 0;
    size_t _mat_rows = 0;
    size_t _mat_cols = 0;
};


class DevNullWriter : public DataWriter {
    virtual void write(const std::vector<TrainDatum> &) {}
};

}
#endif //DEEP_LOCALIZER_DATAWRITER_H
