# ArduinoTensorFlowLite

## A TensorFlow Lite Micro Library in Arduino Style

This library simplifies the use of **TensorFlow Lite Micro** on Arduino boards, offering APIs in the typical _Arduino style_. It avoids the use of _pointers_ or other C++ syntactic constructs that are discouraged within an Arduino sketch.
It was inspired in large part by [ArduTFLite](https://github.com/spaziochirale/ArduTFLite) but includes the latest tensor flow code. It is more geared towards those who require Arduino style APIs and who wish to learn about the process of deploying **TensorFlow** models on constrained edge devices. It provides a number of functions that provide insight into the process of deploying models and these functions are particularly useful in debugging issues with models.

It can work with quantized data or raw float values, detecting the appropriate processing depending on the models meta data.

**ArduinoTFLite** is designed to enable experimentation with **Tiny Machine Learning** on Arduino boards with constrained resources, such as **Arduino Nano 33 BLE**, **Arduino Nano ESP32**, **Arduino Nicla**, **Arduino Portenta**, **ESP32 based devices** and **Arduino Giga R1 WiFi**. Usage is simple and straightforward and You don't need an extensive TensorFlow expertise to code your sketches and the library provides an extensive API that allows you explore the internal meta data of the model.

## Architecture

ArduinoTensorFlowLite consists of an Arduino style abstraction called **ArduinoTFLite** and a port of TensorFlow Lite for Arduino type boards.

## Installation

To install the in-development version of this library, you can use the latest version directly from the [GitHub repository](https://github.com/johnosbb/ArduinoTensorFlowLite). This requires you clone the repo into the folder that holds libraries for the Arduino IDE.

Once you're in that folder in the terminal, you can then grab the code using the git command line tool:

```
git clone https://github.com/johnosbb/ArduinoTensorFlowLite ArduinoTensorFlowLite
```

To update your clone of the repository to the latest code, use the following terminal commands:

```
cd ArduinoTensorFlowLite
git pull
```

## Checking your Installation

Once the library has been installed, you should then start the Arduino IDE. You will now see an `ArduinoTensorFlowLite` entry in the `File -> Examples` menu of the Arduino IDE. This submenu contains a list of sample projects you can try out. These examples show the abstraction layer in use.

## Usage examples

The examples included with the library show how to use the library. The examples come with their pre-trained models.

## General TinyML development process:

1. **Create an Arduino Sketch to collect data suitable for training**: First, create an Arduino sketch to collect data to be used as the training dataset.
2. **Define a DNN model**: Once the training dataset is acquired, create a neural network model using an external TensorFlow development environment, such as Google Colaboratory.
3. **Train the Model**: Import training data on the TensorFlow development environment and train the model on the training dataset.
4. **Convert and Save the Model**: Convert the trained model to TensorFlow Lite format and save it as a `model.h` file. This file should contain the definition of a static byte array corresponding to the binary format (flat buffer) of the TensorFlow Lite model.
5. **Prepare a new Arduino Sketch for Inference**
6. **Include Necessary Headers**: Include the `ArduinoTFLite.h` header file and the `model.h` file.
7. **Define Tensor Arena**: Define a globally sized byte array for the area called tensorArena.
8. **Initialize the Model**: Initialize the model using the `ModelInit()` function.
9. **Set Input Data**: Insert the input data into the model's input tensor using the `ModelSetInput()` function.
10. **Run Inference**: Invoke the inference operation using the `ModelRunInference()` function.
11. **Read Output Data**: Read the output data using the `ModelGetOutput()` function.

### GitHub

The officially supported TensorFlow Lite Micro library for Arduino resides in the [tflite-micro-arduino-examples](https://github.com/tensorflow/tflite-micro-arduino-examples). This library is a fork of that project with the necessary refactoring required to allow the code build in an Arduino IDE environment. The latest version of this library can be found in the repository [ArduinoTensorFlowLite](https://github.com/johnosbb/ArduinoTensorFlowLite)

## Compatibility

This library is designed for the mbed based `Raspberry Pi Pico` and for a range of `ESP32 based boards` boards. The framework code for running machine learning models should be compatible with most Arm Cortex M-based boards.

## License

This code is made available under the Apache 2 license.

## Contributing

Forks of this library are welcome and encouraged. If you have bug reports or fixes to contribute, the source of this code is at [https://github.com/johnosbb/ArduinoTensorFlowLite](https://github.com/johnosbb/ArduinoTensorFlowLite) and all issues and pull requests should be directed there.

## ArduinoTFLite Library Documentation

This section provides a detailed description of the ArduinoTFLite abstraction layer.

### **Function**

**bool ModelInit(const unsigned char* model, byte* tensorArena, int tensorArenaSize);**

_Initializes TensorFlow Lite Micro environment, instantiates an `AllOpsResolver`, and allocates input and output tensors._

**Parameters:**

- `model`: A pointer to the model data.
- `tensorArena`: A memory buffer to be used for tensor allocations.
- `tensorArenaSize`: The size of the tensor arena.

**Returns:**  
`true` = success, `false` = failure.

---

### **Function**

**bool ModelSetInput(float inputValue, int index, bool showQuantizedValue = false);**

_Writes `inputValue` in the position `index` of the input tensor, automatically handling quantization if the tensor is of type `int8`._

**Parameters:**

- `inputValue`: The input value to be written to the tensor.
- `index`: The position in the input tensor where the value is written.
- `showQuantizedValue`: (Optional) If `true`, prints the quantized value for debugging purposes.

**Returns:**  
`true` = success, `false` = failure.

---

### **Function**

**bool ModelRunInference();**

_Invokes the TensorFlow Lite Micro Interpreter and executes the inference algorithm._

**Returns:**  
`true` = success, `false` = failure.

---

### **Function**

**float ModelGetOutput(int index);**

_Returns the output value from the position `index` of the output tensor, automatically handling dequantization if the tensor is of type `int8`._

**Parameters:**

- `index`: The position in the output tensor from which to retrieve the result.

**Returns:**  
The output value from the tensor, or `-1` if there was an error.

---

### **Function**

**void ModelPrintInputTensorDimensions();**

_Prints the dimensions of the input tensor._

**Description:**  
Prints the size and dimensionality of the input tensor to the Serial Monitor, allowing the user to check how the input data should be structured.

---

### **Function**

**void ModelPrintOutputTensorDimensions();**

_Prints the dimensions of the output tensor._

**Description:**  
Prints the size and dimensionality of the output tensor to the Serial Monitor, providing insight into the model’s output shape.

---

### **Function**

**void ModelPrintTensorQuantizationParams();**

_Prints the quantization parameters for both the input and output tensors._

**Description:**  
Prints the scale and zero-point values for the input and output tensors, which are crucial for understanding how floating-point values are converted to and from quantized integer values.

---

### **Function**

**void ModelPrintMetadata();**

_Prints metadata information about the model, including description and version._

**Description:**  
If the model contains a description, it will be printed along with the model version. If no description is available, it will indicate so.

---

### **Function**

**void ModelPrintTensorInfo();**

_Prints detailed information about the input and output tensors, including their types and dimensions._

**Description:**  
Prints the data type (e.g., `int8` or `float32`) and the size of each dimension for both the input and output tensors.

---

### **Helper Functions**

#### **Quantization Functions**

**int8_t QuantizeInput(float x, float scale, float zeroPoint);**

_Quantizes a floating-point value to an `int8_t` value based on the provided scale and zero point._

**Parameters:**

- `x`: The floating-point input value.
- `scale`: The quantization scale.
- `zeroPoint`: The quantization zero point.

**Returns:**  
The quantized `int8_t` value.

---

**float DequantizeOutput(int8_t quantizedValue, float scale, float zeroPoint);**

_Dequantizes an `int8_t` output value back to a floating-point value based on the provided scale and zero point._

**Parameters:**

- `quantizedValue`: The quantized `int8_t` output value.
- `scale`: The quantization scale.
- `zeroPoint`: The quantization zero point.

**Returns:**  
The dequantized floating-point value.

---

### GitHub

The officially supported TensorFlow Lite Micro library for Arduino resides in the [tflite-micro-arduino-examples](https://github.com/tensorflow/tflite-micro-arduino-examples). This library is a fork of that project with the necessary refactoring required to allow the code build in an Arduino IDE environment. The latest version of this library can be found in the repository [ArduinoTensorFlowLite](https://github.com/johnosbb/ArduinoTensorFlowLite)

To install the in-development version of this library, you can use the latest version directly from the GitHub repository. This requires you clone the repo into the folder that holds libraries for the Arduino IDE.

Once you're in that folder in the terminal, you can then grab the code using the git command line tool:

```
git clone https://github.com/johnosbb/ArduinoTensorFlowLite ArduinoTensorFlowLite
```

To update your clone of the repository to the latest code, use the following terminal commands:

```
cd ArduinoTensorFlowLite
git pull
```

## Compatibility

This library is designed for the mbed based `Raspberry Pi Pico` and for a range of `ESP32 based boards` boards. The framework code for running machine learning models should be compatible with most Arm Cortex M-based boards.

## License

This code is made available under the Apache 2 license.

## Contributing

Forks of this library are welcome and encouraged. If you have bug reports or fixes to contribute, the source of this code is at [https://github.com/johnosbb/ArduinoTensorFlowLite](https://github.com/johnosbb/ArduinoTensorFlowLite) and all issues and pull requests should be directed there.
