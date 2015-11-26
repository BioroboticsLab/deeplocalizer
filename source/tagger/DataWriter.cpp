
#include "DataWriter.h"
#include "utils.h"

#include "hdf5.h"
#include "hdf5_hl.h"

namespace deeplocalizer {


namespace io = boost::filesystem;

std::unique_ptr<DataWriter> DataWriter::fromSaveFormat(
        const std::string &output_dir, Dataset::Format save_format) {
    switch (save_format) {
        case Dataset::Format::HDF5:
            return std::make_unique<HDF5Writer>(output_dir);

        case Dataset::Format::Images:
            return std::make_unique<ImageWriter>(output_dir);
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
        io::path image_path = filename(d);
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

        *data_ptr = static_cast<float>(data.taginess());
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
    if (io::exists(hdf5file)) {
        std::stringstream ss;
        ss << "File " << hdf5file << " allready exists. Please use another directory.";
        throw std::invalid_argument(ss.str());
    }
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
