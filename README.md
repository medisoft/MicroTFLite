# TensorFlow Lite Micro Library for Arduino (Fork)

This repository has the code (including examples) needed to use Tensorflow Lite Micro on an Arduino. It features an abstraction layer for the Arduino that encapsulates much of the TensorFlow detail behind an Arduino Style set of APIs.






## How to Install

### GitHub

The officially supported TensorFlow Lite Micro library for Arduino resides
in the [tflite-micro-arduino-examples](https://github.com/tensorflow/tflite-micro-arduino-examples). This library is a fork of that project with the necessary refactoring required to allow the code build in an Arduino IDE environment.

GitHub repository.
To install the in-development version of this library, you can use the latest version directly from the GitHub repository. This requires you clone the repo into the folder that holds libraries for the Arduino IDE. 

Once you're in that folder in the terminal, you can then grab the code using the
git command line tool:

```
git clone https://github.com/tensorflow/tflite-micro-arduino-examples Arduino_TensorFlowLite
```

To update your clone of the repository to the latest code, use the following terminal commands:
```
cd Arduino_TensorFlowLite
git pull
```

### Checking your Installation

Once the library has been installed, you should then start the Arduino IDE. You will now see an `ArduinoTensorFlowLite` entry in the `File -> Examples` menu of the Arduino IDE. This submenu contains a list of sample projects you can try out. These examples show the abstraction layer in use.



## Compatibility

This library is designed for the mbed based `Raspberry Pi Pico` and for a range of `ESP32 based boards` boards. The framework code for running machine learning models should be compatible with most Arm Cortex M-based boards.

## License

This code is made available under the Apache 2 license.

## Contributing

Forks of this library are welcome and encouraged. If you have bug reports or fixes to contribute, the source of this code is at [https://github.com/tensorflow/tflite-micro](http://github.com/tensorflow/tflite-micro) and all issues and pull requests should be directed there.
