#ifndef DEEP_LOCALIZER_DATAWRITER_H
#define DEEP_LOCALIZER_DATAWRITER_H


#include <memory>
#include <boost/filesystem.hpp>
#include <mutex>
#include <lmdb.h>

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
protected:
    inline std::pair<std::string, int> toFilenameLabel(const TrainDatum & datum) {
        return std::make_pair(datum.filename(), datum.tag().isTag());
    }
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
};


class LMDBWriter : public DataWriter {
public:
    LMDBWriter(const std::string &output_dir);
    virtual void write(const std::vector<TrainDatum> &dataset);
    virtual ~LMDBWriter();

private:
    boost::filesystem::path _output_dir;
    MDB_env *_mdb_env;
    std::mutex _mutex;
    unsigned long _id = 0;
    void openDatabase(const boost::filesystem::path &lmdb_dir,
                      MDB_env **mdb_env);
};
class AllFormatWriter : public DataWriter {
public:
    AllFormatWriter(const std::string &output_dir);
    virtual void write(const std::vector<TrainDatum> &dataset);

private:
    std::unique_ptr<LMDBWriter> _lmdb_writer;
    std::unique_ptr<ImageWriter> _image_writer;
};

class DevNullWriter : public DataWriter {
    virtual void write(const std::vector<TrainDatum> &) {}
};

}
#endif //DEEP_LOCALIZER_DATAWRITER_H
