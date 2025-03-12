#ifndef PROGRESS_BAR_HPP
#define PROGRESS_BAR_HPP

#include <string>
#include <chrono>

class ProgressBar {
public:
    ProgressBar(int total, int width = 50, const std::string& prefix = "", const std::string& suffix = "");

    ~ProgressBar();

    void step();

    void update(int current);

    void finish();

private:
    int totalSteps;
    int currentStep;
    int barWidth;
    std::string prefixText;
    std::string suffixText;
    std::chrono::time_point<std::chrono::steady_clock> startTime;

    void display();
};

#endif // PROGRESS_BAR_HPP
