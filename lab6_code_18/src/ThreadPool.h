#pragma once

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

namespace ML {

class ThreadPool {
public:
    static ThreadPool& instance() {
        static ThreadPool pool;
        return pool;
    }

    void parallelFor(std::size_t count, std::function<void(std::size_t, std::size_t)> func) {
        if (count == 0) return;

        unsigned int n = numThreads;
        if (n > count) n = (unsigned int)count;

        if (n <= 1) {
            func(0, count);
            return;
        }

        // Compute chunk boundaries
        std::size_t base = count / n;
        std::size_t extra = count % n;
        boundaries[0] = 0;
        for (unsigned int i = 0; i < n; i++) {
            boundaries[i + 1] = boundaries[i] + base + (i < extra ? 1 : 0);
        }

        // Publish task and wake workers
        {
            std::lock_guard<std::mutex> lock(mtx);
            currentFunc = func;
            numChunks = n;
            finishedCount = 0;
            generation++;
        }
        startCV.notify_all();

        // Main thread does chunk 0
        func(boundaries[0], boundaries[1]);

        // Wait for all worker chunks to complete
        {
            std::unique_lock<std::mutex> lock(mtx);
            doneCV.wait(lock, [this, n]{ return finishedCount >= n - 1; });
        }
    }

    unsigned int getNumThreads() const { return numThreads; }

private:
    ThreadPool()
        : numChunks(0), finishedCount(0), generation(0), shutdown(false) {
        numThreads = std::thread::hardware_concurrency();
        if (numThreads == 0) numThreads = 4;

        for (unsigned int i = 1; i < numThreads; i++) {
            workers.emplace_back([this, i]{ workerLoop(i); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            shutdown = true;
            generation++;
        }
        startCV.notify_all();
        for (std::size_t i = 0; i < workers.size(); i++) workers[i].join();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void workerLoop(unsigned int id) {
        unsigned int myGen = 0;

        while (true) {
            // Wait for a new generation of work
            {
                std::unique_lock<std::mutex> lock(mtx);
                startCV.wait(lock, [this, &myGen]{
                    return generation != myGen || shutdown;
                });
                if (shutdown) return;
                myGen = generation;
            }

            // Do work if this worker has a chunk
            if (id < numChunks) {
                currentFunc(boundaries[id], boundaries[id + 1]);

                // Signal completion only if we did work
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    finishedCount++;
                }
                doneCV.notify_one();
            }
        }
    }

    unsigned int numThreads;
    std::vector<std::thread> workers;

    std::mutex mtx;
    std::condition_variable startCV;
    std::condition_variable doneCV;

    std::function<void(std::size_t, std::size_t)> currentFunc;
    std::size_t boundaries[128];
    unsigned int numChunks;
    unsigned int finishedCount;
    unsigned int generation;
    bool shutdown;
};

} // namespace ML
