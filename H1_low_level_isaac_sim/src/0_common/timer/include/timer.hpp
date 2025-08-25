#ifndef TIMER_H
#define TIMER_H

// #define TIMEIT
#ifdef TIMEIT

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <unordered_map>
#include <numeric>
#include <mutex>
#include <filesystem>
#include <unistd.h>
#include <cstring>



class MultiTaskTimer {
public:
    ~MultiTaskTimer() {
        std::string executable_name = getExecutableName();
        std::string cwd = std::filesystem::current_path().string();
        std::string folder_path = cwd + "/average_timings";
        
        // Check if the folder exists, and create it if it doesn't
        if (!std::filesystem::exists(folder_path)) {
            std::filesystem::create_directory(folder_path);
            std::cout << "Directory 'average_timings' created!" << std::endl;
        }
        
        std::string timestamp = this->getCurrentTimeFormatted();
        std::string file_path = folder_path + "/" + executable_name + "_average_timings_" + timestamp + ".txt";
        saveAverageTimingsToFile(file_path);
        std::cout << "Average timings saved!" << std::endl;
    }

    // Start timing a task
    void start(const std::string& task_name) {
        std::lock_guard<std::mutex> lock(mutex_);
        start_times_[task_name] = std::chrono::high_resolution_clock::now();
    }

    // Stop timing a task and store the duration
    void stop(const std::string& task_name) {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = start_times_.find(task_name);
        if (it != start_times_.end()) {
            double duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - it->second).count();
            cumulative_timings_[task_name] += duration;
            counts_[task_name] += 1;
            max_times_[task_name] = std::max(max_times_[task_name], duration);
            start_times_.erase(it);
        } else {
            std::cerr << "Warning: Task \"" << task_name << "\" was not started." << std::endl;
        }
    }

    // Reset cumulative timings (e.g., after averaging)
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        cumulative_timings_.clear();
        counts_.clear();
    }

    // Print the average timings as a table
    void printAverageTimings() const {
        printAverageAndMaxTimingTable(cumulative_timings_, counts_, max_times_);
    }

    // Save the average timings as a table to a file
    void saveAverageTimingsToFile(const std::string& file_path) const {
        std::ofstream file(file_path, std::ios::trunc);
        if (file.is_open()) {
            printAverageAndMaxTimingTable(cumulative_timings_, counts_, max_times_, file);
            file.close();
        } else {
            std::cerr << "Error: Unable to open file \"" << file_path << "\" for writing." << std::endl;
        }
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> start_times_;
    std::unordered_map<std::string, double> cumulative_timings_;
    std::unordered_map<std::string, int> counts_;
    std::unordered_map<std::string, double> max_times_;

    // Helper to print the average and maximum times table (console or file)
    void printAverageAndMaxTimingTable(
        const std::unordered_map<std::string, double>& cumulative_timings,
        const std::unordered_map<std::string, int>& counts,
        const std::unordered_map<std::string, double>& max_times,
        std::ostream& os = std::cout
    ) const {
        // Print table header
        os << std::left << std::setw(30) << "Step"
        << std::right << std::setw(20) << "Average Time (ms)"
        << std::right << std::setw(20) << "Max Time (ms)" << std::endl;
        os << std::string(70, '-') << std::endl;

        double total_average_time = 0.0;
        double global_max_time = 0.0;
        int valid_steps = 0;

        // Iterate through each step and calculate the average and max times
        for (const auto& [step, total_time] : cumulative_timings) {
            int count = counts.at(step);
            double average_time = (count > 0) ? (total_time / count) / 1000 : 0.0;

            // Retrieve the max time for this step from the max_times_ map
            double max_time = (max_times.count(step) > 0) ? max_times.at(step) / 1000 : 0.0;

            os << std::left << std::setw(30) << step
            << std::right << std::setw(20) << std::fixed << std::setprecision(3) << average_time
            << std::right << std::setw(20) << std::fixed << std::setprecision(3) << max_time << std::endl;

            total_average_time += average_time;
            global_max_time += max_time;
            valid_steps++;
        }

        os << std::string(70, '-') << std::endl;
        os << std::left << std::setw(30) << "Total Average Time"
        << std::right << std::setw(20) << std::fixed << std::setprecision(3) << total_average_time
        << std::right << std::setw(20) << std::fixed << std::setprecision(3) << global_max_time << std::endl;
    }

    std::string getExecutableName() {
        char path[1024];
        ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
        if (len == -1) {
            perror("readlink");
            return "unknown_executable";
        }
        path[len] = '\0'; // Null-terminate the string
        std::filesystem::path exec_path(path);
        return exec_path.stem().string(); // Get the executable name (without path)
    }

    std::string getCurrentTimeFormatted() {
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
            std::tm local_time;
        #ifdef _WIN32
            localtime_s(&local_time, &now_c); // Windows
        #else
            localtime_r(&now_c, &local_time); // Linux/Unix
        #endif

        // Format the time
        std::ostringstream oss;
        oss << std::setw(2) << std::setfill('0') << local_time.tm_mday << "_"
            << std::setw(2) << std::setfill('0') << (local_time.tm_mon + 1) << "_"
            << std::setw(2) << std::setfill('0') << (local_time.tm_year % 100) << "_"
            << std::setw(2) << std::setfill('0') << local_time.tm_hour
            << std::setw(2) << std::setfill('0') << local_time.tm_min;

        return oss.str();
    }

};

inline std::unique_ptr<MultiTaskTimer> fun_timer = std::make_unique<MultiTaskTimer>();

#define START_TIMER(task_name) \
    fun_timer->start(task_name);

#define STOP_TIMER(task_name) \
    fun_timer->stop(task_name);

#else // TIMEIT not defined

#define START_TIMER(task_name)
#define STOP_TIMER(task_name)

#endif // TIMEIT

#endif // TIMER_H
