#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "Config.h"
#include "Model.h"
#include "Profiler.h"
#include "Types.h"
#include "Utils.h"
#include "layers/Convolutional.h"
#include "layers/ConvPool.h"
#include "layers/Dense.h"
#include "layers/Layer.h"
#include "layers/Flatten.h"
#include "layers/MaxPooling.h"
#include "layers/Softmax.h"

#ifdef ZEDBOARD
#include <file_transfer/file_transfer.h>
#endif

namespace ML {

// Build our ML toy model, I added batching with a param N
Model buildToyModel(const Path modelPath) {
    Model model;
    logInfo("--- Building Toy Model ---");

    // --- Conv 1: L1 ---
    // Input shape: 64x64x3
    // Output shape: 60x60x32
    model.addLayer<ConvolutionalLayer>(
        LayerParams{sizeof(fp32), {64, 64, 3}},                                    // Input Data
        LayerParams{sizeof(fp32), {60, 60, 32}},                                   // Output Data
        LayerParams{sizeof(fp32), {5, 5, 3, 32}, modelPath / "conv1_weights.bin"}, // Weights
        LayerParams{sizeof(fp32), {32}, modelPath / "conv1_biases.bin"}            // Bias
    );

    // --- Fused Conv2+Pool1: L2 ---
    // Input shape: 60x60x32
    // Conv output: 56x56x32, then 2x2 max pool
    // Output shape: 28x28x32
    model.addLayer<ConvPoolLayer>(
        LayerParams{sizeof(fp32), {60, 60, 32}},
        LayerParams{sizeof(fp32), {28, 28, 32}},
        LayerParams{sizeof(fp32), {5, 5, 32, 32}, modelPath / "conv2_weights.bin"},
        LayerParams{sizeof(fp32), {32}, modelPath / "conv2_biases.bin"}
    );

    // --- Conv 3: L4 ---
    // Input shape: 28x28x32
    // Output shape: 26x26x64
    model.addLayer<ConvolutionalLayer>(
        LayerParams{sizeof(fp32), {28, 28, 32}},
        LayerParams{sizeof(fp32), {26, 26, 64}},
        LayerParams{sizeof(fp32), {3, 3, 32, 64}, modelPath / "conv3_weights.bin"},
        LayerParams{sizeof(fp32), {64}, modelPath / "conv3_biases.bin"}
    );

    // --- Fused Conv4+Pool2: L4 ---
    // Input shape: 26x26x64
    // Conv output: 24x24x64, then 2x2 max pool
    // Output shape: 12x12x64
    model.addLayer<ConvPoolLayer>(
        LayerParams{sizeof(fp32), {26, 26, 64}},
        LayerParams{sizeof(fp32), {12, 12, 64}},
        LayerParams{sizeof(fp32), {3, 3, 64, 64}, modelPath / "conv4_weights.bin"},
        LayerParams{sizeof(fp32), {64}, modelPath / "conv4_biases.bin"}
    );

    // --- Conv 5: L7 ---
    // Input shape: 12x12x64
    // Output shape: 10x10x64
    model.addLayer<ConvolutionalLayer>(
        LayerParams{sizeof(fp32), {12, 12, 64}},
        LayerParams{sizeof(fp32), {10, 10, 64}},
        LayerParams{sizeof(fp32), {3, 3, 64, 64}, modelPath / "conv5_weights.bin"},
        LayerParams{sizeof(fp32), {64}, modelPath / "conv5_biases.bin"}
    );

    // --- Fused Conv6+Pool3: L6 ---
    // Input shape: 10x10x64
    // Conv output: 8x8x128, then 2x2 max pool
    // Output shape: 4x4x128
    model.addLayer<ConvPoolLayer>(
        LayerParams{sizeof(fp32), {10, 10, 64}},
        LayerParams{sizeof(fp32), {4, 4, 128}},
        LayerParams{sizeof(fp32), {3, 3, 64, 128}, modelPath / "conv6_weights.bin"},
        LayerParams{sizeof(fp32), {128}, modelPath / "conv6_biases.bin"}
    );

    // --- Flatten 1: L10 ---
    // Input shape: 4x4x128
    // Output shape: 2048
    model.addLayer<FlattenLayer>(
        LayerParams{sizeof(fp32), {4, 4, 128}},
        LayerParams{sizeof(fp32), {2048}}
    );

    // --- Dense 1: L11 ---
    // Input shape: 2048
    // Output shape: 256
    model.addLayer<DenseLayer>(
        LayerParams{sizeof(fp32), {2048}},
        LayerParams{sizeof(fp32), {256}},
        LayerParams{sizeof(fp32), {2048, 256}, modelPath / "dense1_weights.bin"},
        LayerParams{sizeof(fp32), {256}, modelPath / "dense1_biases.bin"}
    );

    // --- Dense 2: L12 ---
    // Input shape: 256
    // Output shape: 200
    model.addLayer<DenseLayer>(
        LayerParams{sizeof(fp32), {256}},
        LayerParams{sizeof(fp32), {200}},
        LayerParams{sizeof(fp32), {256, 200}, modelPath / "dense2_weights.bin"},
        LayerParams{sizeof(fp32), {200}, modelPath / "dense2_biases.bin"},
        ActivationType::NONE
    );

    // --- Softmax 1: L13 ---
    // Input shape: 200
    // Output shape: 200
    model.addLayer<SoftmaxLayer>(
        LayerParams{sizeof(fp32), {200}},
        LayerParams{sizeof(fp32), {200}}
    );

    return model;
}

void runBasicTest(Model& model, const Path& basePath) {
    logInfo("--- Running Basic Test ---");

    // Load an image
    LayerData img = {{sizeof(fp32), {64, 64, 3}, "./data/image_0.bin"}};
    img.loadData();

    // Compare images
    std::cout << "Comparing image 0 to itself (max error): " << img.compare<fp32>(img) << std::endl
              << "Comparing image 0 to itself (T/F within epsilon " << ML::Config::EPSILON << "): " << std::boolalpha
              << img.compareWithin<fp32>(img, ML::Config::EPSILON) << std::endl;

    // Test again with a modified copy
    std::cout << "\nChange a value by 0.1 and compare again" << std::endl;
    
    LayerData imgCopy = img;
    imgCopy.get<fp32>(0) += 0.1;

    // Compare images
    img.compareWithinPrint<fp32>(imgCopy);

    // Test again with a modified copy
    log("Change a value by 0.1 and compare again...");
    imgCopy.get<fp32>(0) += 0.1;

    // Compare Images
    img.compareWithinPrint<fp32>(imgCopy);
    std::cout << "\n" << std::endl;
}

void runLayerTest(const std::size_t layerNum, Model& model, const Path& basePath) {
    logInfo(std::string("--- Running Layer Test ") + std::to_string(layerNum) + " ---");

    if (layerNum >= model.getNumLayers()) {
        throw std::runtime_error(
            "Invalid layer number"
        );
    }

    LayerData image(
        model[0].getInputParams(),
        basePath / "image_0.bin"
    );
    image.loadData();

    const LayerData* currentInput = &image;

    const LayerData* output = nullptr;

    Timer timer("Layer Inference");
    timer.start();


    for (std::size_t i = 0; i <= layerNum; i++) {
        output = &model.inferenceLayer(
            *currentInput,
            i,
            Layer::InfType::NAIVE
        );

        currentInput = output;
    }

    timer.stop();

    Path expectedPath =
        basePath / "image_0_data" /
        ("layer_" + std::to_string(layerNum) + "_output.bin");

    LayerData expected(
        output->getParams(),
        expectedPath
    );

    expected.loadData();

    bool passed = output->compareWithinPrint<fp32>(expected);

    std::cout
        << "Layer " << layerNum
        << " verification: "
        << (passed ? "PASS" : "FAIL")
        << "\n"
        << std::endl;
}

void runInferenceTest(Model& model, const Path& basePath, Layer::InfType infType = Layer::InfType::NAIVE, const std::string& label = "NAIVE") {
    logInfo(std::string("--- Running Inference Test (") + label + ") ---");

    LayerData img(model[0].getInputParams(), basePath / "image_0.bin");
    img.loadData();

    Profiler profiler(model);

    const LayerData& output = model.inference(img, infType, &profiler);

    LayerData expected(model.getOutputLayer().getOutputParams(), basePath / "image_0_data" / "layer_11_output.bin");
    expected.loadData();
    output.compareWithinPrint<fp32>(expected);
    std::cout << "\n" << std::endl;

    profiler.printReport();
}

void runValidationEval(Model& model, int numImages) {
    std::ifstream labelsFile("data/val_images/labels.txt");
    if (!labelsFile.is_open()) {
        logError("failed labels.txt");
        return;
    }
    std::vector<int> labels(numImages);
    for (int i = 0; i < numImages; i++) {
        int idx, lbl;
        labelsFile >> idx >> lbl;
        labels[i] = lbl;
    }
    labelsFile.close();

    int top1_correct = 0;
    int top10_correct = 0;
    float total_time_ms = 0;

    for (int img = 0; img < numImages; img++) {
        std::string imgFile = "data/val_images/image_" + std::to_string(img) + ".bin";
        LayerData image(model[0].getInputParams(), Path(imgFile.c_str()));
        image.loadData();

        Timer timer("img");
        timer.start();
        const LayerData& output = model.inference(image, Layer::InfType::NAIVE);
        timer.stop();
        total_time_ms += timer.milliseconds;

        size numClasses = output.getParams().flat_count();
        std::vector<std::pair<fp32, int>> preds(numClasses);
        for (size c = 0; c < numClasses; c++) {
            preds[c] = {output.get<fp32>(c), (int)c};
        }
        std::sort(preds.begin(), preds.end(), [](const std::pair<fp32,int>& a, const std::pair<fp32,int>& b) {
            return a.first > b.first;
        });

        if (preds[0].second == labels[img]) {
            top1_correct++;
        }
        for (int k = 0; k < 10 && k < (int)numClasses; k++) {
            if (preds[k].second == labels[img]) {
                top10_correct++;
                break;
            }
        }

        if ((img + 1) % 100 == 0) {
            std::cout << "  " << (img + 1) << "/" << numImages << " done" << std::endl;
        }
    }

    float top1_acc = (float)top1_correct / numImages * 100.0f;
    float top10_acc = (float)top10_correct / numImages * 100.0f;
    float avg_latency = total_time_ms / numImages;

    std::cout << "\n" << std::endl;
    std::cout << "Top-1 Accuracy: " << top1_correct << "/" << numImages << " = " << top1_acc << "%" << std::endl;
    std::cout << "Top-10 Accuracy: " << top10_correct << "/" << numImages << " = " << top10_acc << "%" << std::endl;
    std::cout << "Avg latency per image: " << avg_latency << " ms" << std::endl;
}

void runTests() {
    // Base input data path (determined from current directory of where you are running the command)
    Path basePath("data");  // May need to be altered for zedboards loading from SD Cards

    // Build the model and allocate the buffers
    Model model = buildToyModel(basePath / "model");
    model.allocLayers();

    // Run some framework tests as an example of loading data
    runBasicTest(model, basePath);

    // Run a layer inference test
    runLayerTest(0, model, basePath);

    // Run end-to-end inference tests
    runInferenceTest(model, basePath, Layer::InfType::NAIVE, "NAIVE");
    runInferenceTest(model, basePath, Layer::InfType::TILED, "TILED");
    runInferenceTest(model, basePath, Layer::InfType::THREADED, "THREADED");
    runInferenceTest(model, basePath, Layer::InfType::COMBINED, "TILED+THREADED");

    // We made this function to test on multiple images
    // runValidationEval(model, 1000);

    // Clean up
    model.freeLayers();
    std::cout << "\n\n----- ML::runTests() COMPLETE -----\n";
}

} // namespace ML

#ifdef ZEDBOARD
extern "C"
int main() {
    try {
        static FATFS fatfs;
        if (f_mount(&fatfs, "/", 1) != FR_OK) {
            throw std::runtime_error("Failed to mount SD card. Is it plugged in?");
        }
        ML::runTests();
    } catch (const std::exception& e) {
        std::cerr << "\n\n----- EXCEPTION THROWN -----\n" << e.what() << '\n';
    }
    std::cout << "\n\n----- STARTING FILE TRANSFER SERVER -----\n";
    FileServer::start_file_transfer_server();
}
#else
int main() {
    ML::runTests();
}
#endif
