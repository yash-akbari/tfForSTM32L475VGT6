#include "mbed.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Include the generated model header
#include "afLSTM.h" // Ensure this is the header file where your model data is defined

// Define constants for the model input size and tensor pool
constexpr int kInputSize = 40; // Adjust based on your model input size
constexpr int kTensorPoolSize = 32 * 1024; // Allocate 32 KB for tensor pool
alignas(16) uint8_t tensor_pool[kTensorPoolSize];

// Sample RR interval data (replace with actual data)
uint8_t rr_intervals[kInputSize] = {
    174, 174, 100, 175, 114, 145, 165, 200, 140, 123,
    115, 161, 126, 130, 170, 221, 205, 179, 140, 101,
    125, 109, 114, 176, 148, 181, 164, 248, 126, 137,
    218, 181, 117, 129, 117, 106, 122, 125, 156, 162
};

// Function to log debug messages
void DebugLog(const char* message) {
    printf("%s\n", message);
}

// Function to check and log tensor details
void LogTensorDetails(const TfLiteTensor* tensor, int index) {
    if (tensor == nullptr) {
        printf("Tensor %d is null\n", index);
        return;
    }
    printf("Tensor %d - Type: %d, Size: %d, Dimensions: [", index, tensor->type, tensor->bytes);
    for (int i = 0; i < tensor->dims->size; i++) {
        printf("%d", tensor->dims->data[i]);
        if (i < tensor->dims->size - 1) printf(", ");
    }
    printf("]\n");
}

// Main function with detailed logging
int main() {
    DebugLog("Starting AFib Detection...");

    // Load the TFLite model
    DebugLog("Retrieving model pointer...");
    const tflite::Model* model = tflite::GetModel(afLSTM);
    if (model == nullptr) {
        DebugLog("Failed to retrieve model pointer.");
        return 1;
    }
    DebugLog("Model pointer retrieved successfully.");

    // Check model version
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        DebugLog("Model schema version mismatch.");
        return 1;
    }
    DebugLog("Model schema version verified.");

    // Initialize the error reporter
    DebugLog("Initializing error reporter...");
    tflite::MicroErrorReporter error_reporter;
    DebugLog("Error reporter initialized.");

    // Set up the TensorFlow Lite Micro interpreter
    DebugLog("Setting up interpreter...");
    tflite::ops::micro::AllOpsResolver resolver;
    tflite::MicroInterpreter interpreter(model, resolver, tensor_pool, kTensorPoolSize, &error_reporter);
    DebugLog("Interpreter setup completed.");

    // Allocate tensors and check status
    DebugLog("Allocating tensors...");
    TfLiteStatus allocate_status = interpreter.AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        DebugLog("Tensor allocation failed. Checking tensor details:");
        // Log all tensors to diagnose
        for (int i = 0; i < interpreter.inputs_size(); i++) {
            LogTensorDetails(interpreter.input(i), i);
        }
        for (int i = 0; i < interpreter.outputs_size(); i++) {
            LogTensorDetails(interpreter.output(i), i);
        }
        return 1;
    }
    DebugLog("Tensor allocation successful.");

    // Get a pointer to the input tensor and copy the RR intervals into it
    float* input = interpreter.input(0)->data.f; // Assuming the input type is float
    DebugLog("Copying RR interval data to model input...");
    for (int i = 0; i < kInputSize; i++) {
        input[i] = static_cast<float>(rr_intervals[i]); // Convert uint8 to float if needed
    }
    DebugLog("RR interval data copied to model input.");

    // Run inference on the model
    DebugLog("Running inference...");
    TfLiteStatus invoke_status = interpreter.Invoke();
    if (invoke_status != kTfLiteOk) {
        DebugLog("Error during model invocation.");
        return 1;
    }
    DebugLog("Inference completed successfully.");

    // Get the output from the model
    float afib_score = interpreter.output(0)->data.f[0];
    DebugLog("AFib score retrieved from model output.");

    // Interpret the result and print the score
    if (afib_score > 0.5f) {
        DebugLog("AFib detected!");
    } else {
        DebugLog("Normal rhythm detected.");
    }
    printf("AFib Score: %.2f\n", afib_score);

    // Loop indefinitely
    while (true) {
        // Continuous processing or sleep for power-saving
    }

    return 0;
}