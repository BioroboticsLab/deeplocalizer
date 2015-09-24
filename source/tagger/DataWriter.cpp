
#include "DataWriter.h"
#include "utils.h"

#include "hdf5.h"
#include "hdf5_hl.h"

namespace deeplocalizer {


namespace io = boost::filesystem;

std::unique_ptr<DataWriter> DataWriter::fromSaveFormat(
        const std::string &output_dir, Dataset::Format save_format) {
    switch (save_format) {
        case Dataset::Format::All:
            return std::make_unique<AllFormatWriter>(output_dir);
        case Dataset::Format::HDF5:
            return std::make_unique<HDF5Writer>(output_dir);
        case Dataset::Format::Images:
            return std::make_unique<ImageWriter>(output_dir);
        case Dataset::Format::LMDB:
            return std::make_unique<LMDBWriter>(output_dir);
        default:
            return std::make_unique<DevNullWriter>();
    }
}

ImageWriter::ImageWriter(const std::string &output_dir)
    : _output_dir{output_dir}
{
    io::create_directories(_output_dir);
    io::path txt_file = _output_dir / _output_dir.filename();
    txt_file.replace_extension(".txt");
    _stream.open(txt_file.string());
}

void ImageWriter::write(const std::vector<TrainDatum> &data) {
    writeImages(data);
    writeLabelFile(data);
}
void ImageWriter::writeImages(const std::vector<TrainDatum> &data) const {
    for(const auto & d: data) {
        io::path output_file(_output_dir);
        io::path image_path = d.filename();
        output_file /= image_path.filename();
        cv::imwrite(output_file.string(), d.mat());
    }
}

void ImageWriter::writeLabelFile(const std::vector<TrainDatum> &data) {
    std::lock_guard<std::mutex> lock(_mutex);
    for(const auto & datum : data) {
        const auto pair = toFilenameLabel(datum);
        io::path image_path = _output_dir / pair.first;
        _stream << image_path.string() << " " << pair.second << "\n";
    }
}

LMDBWriter::LMDBWriter(const std::string &output_dir) :
    _output_dir(output_dir)
{
    io::create_directories(_output_dir);
    openDatabase(_output_dir, &_mdb_env);
}

void LMDBWriter::openDatabase(const boost::filesystem::path &lmdb_dir,
                                     MDB_env **mdb_env
) {
    ASSERT(mdb_env_create(mdb_env) == MDB_SUCCESS,
           "mdb_env_create failed");
    ASSERT(mdb_env_set_mapsize(*mdb_env, 1099511627776) == MDB_SUCCESS, "");  // 1TB
    ASSERT(mdb_env_open(*mdb_env, lmdb_dir.string().c_str(), 0, 0664) == MDB_SUCCESS,
           "mdb_env_open failed");
}

unsigned long swap(unsigned long i) {
    long b0, b1, b2, b3, b4, b5, b6, b7;
    b0 = (i & 0x00000000000000ff) << 56u;
    b1 = (i & 0x000000000000ff00) << 40u;
    b2 = (i & 0x0000000000ff0000) << 24u;
    b3 = (i & 0x00000000ff000000) << 8u;
    b4 = (i & 0x000000ff00000000) >> 8u;
    b5 = (i & 0x0000ff0000000000) >> 24u;
    b6 = (i & 0x00ff000000000000) >> 40u;
    b7 = (i & 0xff00000000000000) >> 56u;
    return b0 | b1 | b2 | b3 | b4 | b5 | b6 | b7;
}

void LMDBWriter::write(const std::vector<TrainDatum> &data) {
    const size_t n = 1024;
    auto & mdb_env =  _mdb_env;
    auto & mutex = _mutex;
    std::lock_guard<std::mutex> lock(mutex);
    auto indecies = shuffledIndecies(data.size());
    MDB_txn * mdb_txn = nullptr;
    MDB_val mdb_key, mdb_data;
    MDB_dbi mdb_dbi;
    ASSERT(mdb_env != nullptr, "Error: mdb_env is nullptr");

    ASSERT(mdb_txn_begin(mdb_env, nullptr, 0, &mdb_txn) == MDB_SUCCESS,
           "mdb_txn_begin failed");
    ASSERT(mdb_dbi_open(mdb_txn, nullptr, 0, &mdb_dbi) == MDB_SUCCESS,
           "mdb_open failed. Does the lmdb already exist? ");
    std::string data_string;
    for(size_t i = 0; i < data.size(); i++) {
        const auto & d = data.at(indecies.at(i));
        auto caffe_datum = d.toCaffe();
        caffe_datum.SerializeToString(&data_string);
        // lmdb uses memcmp therefore we convert to big endian
        unsigned long key = swap(_id++);
        mdb_data.mv_size = data_string.size();
        mdb_data.mv_data = reinterpret_cast<void *>(&data_string[0]);
        mdb_key.mv_size = sizeof(unsigned long);
        mdb_key.mv_data = reinterpret_cast<void *>(&key);

        auto err = mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, MDB_APPEND);
        if (err != MDB_SUCCESS)
            ASSERT(false, "mdb_put failed: " << err << ", key: " << _id-1);

        if(i % n == 0 && i != 0) {
            ASSERT(mdb_txn_commit(mdb_txn) == MDB_SUCCESS,
                   "mdb_txn_commit failed");
            ASSERT(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn) == MDB_SUCCESS,
                   "mdb_txn_begin failed");
        }
    }
    ASSERT(mdb_txn_commit(mdb_txn) == MDB_SUCCESS, "mdb_txn_commit failed");
    mdb_dbi_close(mdb_env, mdb_dbi);
}

LMDBWriter::~LMDBWriter() {
    mdb_env_close(_mdb_env);
}
AllFormatWriter::AllFormatWriter(const std::string &output_dir) :
    _lmdb_writer(std::make_unique<LMDBWriter>(output_dir)),
    _image_writer(std::make_unique<ImageWriter>(output_dir))
{
}

void AllFormatWriter::write(const std::vector<TrainDatum> &dataset) {
    _lmdb_writer->write(dataset);
    _image_writer->write(dataset);
}

HDF5Writer::HDF5Writer(const std::string &output_dir)
    : _output_dir(output_dir),
      _unique_path_format((_output_dir / _output_dir.filename()).string())
{
    io::create_directories(output_dir);
    _txt_file.open(_unique_path_format + ".txt");
}

void HDF5Writer::saveDataHDF5Dataset(const hid_t file_id) {
    int num_axes = 4;
    const std::string dataset_name = "data";
    size_t nb_imgs = _buffer.size();
    hsize_t dims[] = {nb_imgs, 1, _mat_rows, _mat_cols};
    float * data = static_cast<float*>(malloc(nb_imgs*TAG_HEIGHT*TAG_WIDTH * sizeof(float)));
    float * data_ptr = data;
    for (const TrainDatum & data: _buffer) {
        ASSERT(data.mat().isContinuous(), "mat must be continous");
        std::copy(data.mat().datastart, data.mat().dataend, data_ptr);
        data_ptr += data.mat().dataend - data.mat().datastart;
    }
    herr_t status = H5LTmake_dataset_float(
        file_id, dataset_name.c_str(), num_axes, dims, data);
    ASSERT(status >= 0, "Failed to make float dataset " + dataset_name << ". With status: " << status);
    delete[] data;
}

void HDF5Writer::saveLabelHDF5Dataset(const hid_t file_id) {
    int num_axes = 1;
    const std::string dataset_name = "labels";
    size_t nb_imgs = _buffer.size();
    hsize_t dims[] = {nb_imgs};
    float * data = static_cast<float*>(malloc(nb_imgs * sizeof(float)));
    float * data_ptr = data;
    for (const TrainDatum & data: _buffer) {
        *data_ptr = data.tag().isTag();
        ++data_ptr;
    }
    herr_t status = H5LTmake_dataset_float(
        file_id, dataset_name.c_str(), num_axes, dims, data);
    ASSERT(status >= 0, "Failed to make float dataset " + dataset_name << ". With status: " << status);
    delete[] data;
}


void HDF5Writer::writeBufferToFile()
{
    std::string hdf5file = nextFilename();
    hid_t file_id = H5Fopen(hdf5file.c_str(), H5F_ACC_CREAT | H5F_ACC_RDWR, H5P_DEFAULT);
    saveDataHDF5Dataset(file_id);
    saveLabelHDF5Dataset(file_id);
    _txt_file << io::path(hdf5file).filename().string() << "\n";
    _buffer.clear();
    H5Fclose(file_id);
}

std::string HDF5Writer::nextFilename()
{
    std::stringstream ss;
    ss << _unique_path_format << '_' << _file_id++ << ".hdf5";
    return ss.str();
}

void HDF5Writer::write(const std::vector<TrainDatum> &dataset)
{
    std::lock_guard<std::mutex> lk(_mutex);
    if (_mat_size == 0) {
        _mat_size = dataset.at(0).mat().total();
        _mat_rows = dataset.at(0).mat().rows;
        _mat_cols = dataset.at(0).mat().cols;

        _max_buffer_size = static_cast<size_t>(floor(MAX_HDF5_FILE / (_mat_size * sizeof(float))));
        std::cout << _max_buffer_size << std::endl;
    }
    for(const auto & data : dataset) {
        ASSERT(data.mat().total() == _mat_size,
               "Images in dataset must have same size. Global total size is: "
               << _mat_size << ", but got: " << data.mat().total());
        _buffer.push_back(data);
        if (_buffer.size() >= _max_buffer_size) {
            writeBufferToFile();
        }
    }
}

HDF5Writer::~HDF5Writer()
{
    if (_buffer.size()) {
        writeBufferToFile();
    }
}

}
