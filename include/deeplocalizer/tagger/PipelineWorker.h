#ifndef DEEP_LOCALIZER_PIPELINETHREAD_H
#define DEEP_LOCALIZER_PIPELINETHREAD_H

#include <memory>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <atomic>
#include <boost/filesystem.hpp>
#include <pipeline/Preprocessor.h>
#include <pipeline/Localizer.h>
#include <pipeline/EllipseFitter.h>


namespace deeplocalizer {
class ImageDesc;
class Tag;

class PipelineWorker {
public:
    PipelineWorker();
    PipelineWorker(const std::string &  config_file);
    static const std::string DEFAULT_CONFIG_FILE;

    using tags_proposals_callback_t = std::function<void(ImageDesc)>;
    using find_ellipse_callback_t =  std::function<void(Tag)>;
    using tags_proposals_item_t =  std::pair<ImageDesc, tags_proposals_callback_t>;
    using find_ellipse_item_t =  std::tuple<cv::Mat, Tag, find_ellipse_callback_t>;
    void process(ImageDesc img, std::function<void(ImageDesc)> &&callback);
    void findEllipse(cv::Mat mat, Tag tag,  std::function<void(Tag)> && callback);
    void quit();
private:
    void run();
    void setupPipeline();
    std::vector<Tag> tagsProposals(ImageDesc & img);
    
    inline bool proposalsWorkToDo() {
        std::lock_guard<std::mutex> lk(_mutex_deques);
        return _proposals_todo.size() > 0;
    }

    inline bool findEllipseWorkToDo() {
        std::lock_guard<std::mutex> lk(_mutex_deques);
        return _find_ellipse_todo.size() > 0;
    }
    void processFindEllipse();
    void processTagsProposals();

    boost::filesystem::path _config_file;
    std::unique_ptr<pipeline::Preprocessor> _preprocessor;
    std::unique_ptr<pipeline::Localizer> _localizer;
    std::unique_ptr<pipeline::EllipseFitter> _ellipseFitter;
    static const std::string DEFAULT_JSON_SETTINGS;
    std::condition_variable _work_cv;
    std::mutex _mutex_work_cv;
    std::mutex _mutex_deques;
    std::atomic_bool _quit;
    std::deque<tags_proposals_item_t> _proposals_todo;
    std::deque<find_ellipse_item_t> _find_ellipse_todo;
    std::thread _thread;
};
}


#endif //DEEP_LOCALIZER_PIPELINETHREAD_H
