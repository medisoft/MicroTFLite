#include "MicroTFLite.h"

// TensorFlow Lite components
tflite::AllOpsResolver tflOpsResolver;
const tflite::Model *tflModel = nullptr;
tflite::MicroInterpreter *tflInterpreter = nullptr;
TfLiteTensor *tflInputTensor = nullptr;
TfLiteTensor *tflOutputTensor = nullptr;
float tflInputScale = 0.0f;
int32_t tflInputZeroPoint = 0;
float tflOutputScale = 0.0f;
int32_t tflOutputZeroPoint = 0;

// Initializes the TensorFlow Lite model and interpreter
bool ModelInit(const unsigned char *model, byte *tensorArena, int tensorArenaSize)
{
    tflModel = tflite::GetModel(model);
    if (tflModel->version() != TFLITE_SCHEMA_VERSION)
    {
        Serial.println("Model schema version mismatch!");
        return false;
    }

    tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize);
    tflInterpreter->AllocateTensors();
    tflInputTensor = tflInterpreter->input(0);
    tflOutputTensor = tflInterpreter->output(0);

    // Ensure tensors are initialized
    if (tflInputTensor == nullptr || tflOutputTensor == nullptr)
    {
        Serial.println("Tensors are not initialized.");
        return false;
    }

    // Set scales and zero points. We are using per-tensor quantization to apply the same scale and zero-point to all values in the input (tflInputTensor) and output (tflOutputTensor) tensors.
    tflInputScale = tflInputTensor->params.scale;
    tflInputZeroPoint = tflInputTensor->params.zero_point;
    tflOutputScale = tflOutputTensor->params.scale;
    tflOutputZeroPoint = tflOutputTensor->params.zero_point;
    return true;
}

// Prints metadata about the model, such as description and version
void ModelPrintMetadata()
{
    if (tflModel->description() != nullptr)
    {
        const flatbuffers::String *description = tflModel->description();
        Serial.print("Model Description: ");
        Serial.println(description->str().c_str());
    }
    else
    {
        Serial.println("No model description available.");
    }
    Serial.print("Model Version: ");
    Serial.println(tflModel->version());
}

// Prints information about the tensors, including type and dimensions
void ModelPrintTensorInfo()
{
    if (tflInputTensor == nullptr || tflOutputTensor == nullptr)
    {
        Serial.println("Tensors are not initialized.");
        return;
    }

    // Print input tensor info
    Serial.println("Input Tensor Information:");
    Serial.print("Type: ");
    Serial.println(tflInputTensor->type == kTfLiteFloat32 ? "float32" : "int8");
    Serial.print("Dimensions: ");
    for (int i = 0; i < tflInputTensor->dims->size; ++i)
    {
        Serial.print(tflInputTensor->dims->data[i]);
        if (i < tflInputTensor->dims->size - 1)
            Serial.print(" x ");
    }
    Serial.println();

    // Print output tensor info
    Serial.println("Output Tensor Information:");
    Serial.print("Type: ");
    Serial.println(tflOutputTensor->type == kTfLiteFloat32 ? "float32" : "int8");
    Serial.print("Dimensions: ");
    for (int i = 0; i < tflOutputTensor->dims->size; ++i)
    {
        Serial.print(tflOutputTensor->dims->data[i]);
        if (i < tflOutputTensor->dims->size - 1)
            Serial.print(" x ");
    }
    Serial.println();
}

// Prints the quantization parameters for the input and output tensors
void ModelPrintTensorQuantizationParams()
{
    if (tflInputTensor == nullptr || tflOutputTensor == nullptr)
    {
        Serial.println("Tensors are not initialized.");
        return;
    }

    Serial.println("Input Tensor Quantization Parameters:");
    Serial.print("Scale: ");
    Serial.println(tflInputScale);
    Serial.print("Zero Point: ");
    Serial.println(tflInputZeroPoint);

    Serial.println("Output Tensor Quantization Parameters:");
    Serial.print("Scale: ");
    Serial.println(tflOutputScale);
    Serial.print("Zero Point: ");
    Serial.println(tflOutputZeroPoint);
}

// Quantizes a float value for int8 tensors
// In quantization, floating-point values (e.g., activations or weights in the model) are mapped to integer values (e.g., 8-bit integers) for efficiency.
// The scale and zero-point are used to convert between the floating-point and integer domains.
inline int8_t QuantizeInput(float x, float scale, float zeroPoint)
{
    float quantizedFloat = (x / scale) + zeroPoint;
    int8_t quantizedValue = static_cast<int8_t>(quantizedFloat);
    return quantizedValue;
}

// Dequantizes an int8 value to a float
inline float DequantizeOutput(int8_t quantizedValue, float scale, float zeroPoint)
{
    return ((float)quantizedValue - zeroPoint) * scale;
}

// Sets the input tensor with a given value, handling quantization if needed
bool ModelSetInput(float inputValue, int index, bool showQuantizedValue)
{
    if (tflInputTensor == nullptr || index >= tflInputTensor->dims->data[1])
    {
        Serial.print("Input tensor index out of range!: ");
        Serial.print(index);
        Serial.print(" Range: ");
        Serial.println(tflInputTensor->dims->data[1]);
        return false;
    }

    if (tflInputTensor->type == kTfLiteInt8)
    {
        int8_t quantizedValue = QuantizeInput(inputValue, tflInputScale, tflInputZeroPoint);
        tflInputTensor->data.int8[index] = quantizedValue;
        if (showQuantizedValue)
        {
            Serial.print("Quantized value for index: ");
            Serial.print(index);
            Serial.print(" : ");
            Serial.print(quantizedValue);
            Serial.print(" , input : ");
            Serial.println(inputValue);
        }
    }
    else if (tflInputTensor->type == kTfLiteFloat32)
    {
        tflInputTensor->data.f[index] = inputValue;
    }
    else
    {
        Serial.println("Unsupported input tensor type!");
        return false;
    }

    return true;
}

// Prints the dimensions of the output tensor
void ModelPrintOutputTensorDimensions()
{
    if (tflOutputTensor == nullptr || tflOutputTensor->dims->size == 0)
    {
        Serial.println("Output tensor is null or has no dimensions!");
        return;
    }

    Serial.print("Output tensor dimensions: ");
    Serial.println(tflOutputTensor->dims->size);
    for (int i = 0; i < tflOutputTensor->dims->size; ++i)
    {
        Serial.print("Output Dimension ");
        Serial.print(i);
        Serial.print(": ");
        Serial.println(tflOutputTensor->dims->data[i]);
    }
}

// Prints the dimensions of the input tensor
void ModelPrintInputTensorDimensions()
{
    if (tflInputTensor == nullptr)
    {
        Serial.println("Input tensor is null!");
        return;
    }

    Serial.print("Input tensor dimensions: ");
    Serial.println(tflInputTensor->dims->size);

    for (int i = 0; i < tflInputTensor->dims->size; ++i)
    {
        Serial.print("Dimension ");
        Serial.print(i);
        Serial.print(": ");
        Serial.println(tflInputTensor->dims->data[i]);
    }
}

// Runs inference on the model
bool ModelRunInference()
{
    TfLiteStatus invokeStatus = tflInterpreter->Invoke();
    if (invokeStatus != kTfLiteOk)
    {
        Serial.println("Inference failed!");
        return false;
    }
    return true;
}

// Retrieves the output value from the model
float ModelGetOutput(int index)
{
    if (tflOutputTensor == nullptr || index >= tflOutputTensor->dims->data[1])
    {
        Serial.println("Output tensor index out of range!");
        return -1;
    }

    if (tflOutputTensor->type == kTfLiteInt8)
    {
        int8_t quantizedValue = tflOutputTensor->data.int8[index];
        return DequantizeOutput(quantizedValue, tflOutputScale, tflOutputZeroPoint);
    }
    else if (tflOutputTensor->type == kTfLiteFloat32)
    {
        return tflOutputTensor->data.f[index];
    }
    else
    {
        Serial.println("Unsupported output tensor type!");
        return -1;
    }
}
