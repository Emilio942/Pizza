# GEMINI.md

## Project Overview

This project is a pizza detection system designed for the RP2040 microcontroller. It uses a deep learning model to classify different states of pizza (e.g., raw, cooked, burnt). The project is written in Python and uses PyTorch for model training, OpenCV for image processing, and various other tools for data augmentation, quantization, and deployment.

The project includes an emulator for the RP2040, which allows for testing and development without the need for physical hardware. The emulator also includes a simulated temperature sensor to monitor the device's temperature during operation.

The main goal of the project is to create a lightweight and efficient model that can run on the resource-constrained RP2040 microcontroller.

## Building and Running

### Installation

1.  Create and activate a Python virtual environment:

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Training

There are two ways to train the model:

1.  **Using the main script:**

    ```bash
    python src/pizza_detector.py train
    ```

2.  **Using the training script with augmentation:**

    ```bash
    python scripts/train_with_augmentation.py
    ```

### Exporting the Model

To export the trained model for deployment on the RP2040, run the following command:

```bash
python src/pizza_detector.py export
```

This will generate a C array containing the model weights, which can be compiled into the firmware for the RP2040.

### Running the Emulator

To run the RP2040 emulator, use the following command:

```bash
python src/emulation/emulator.py
```

### Running Tests

To run the temperature logging test, execute the following command:

```bash
python -m tests.test_temperature_logging
```

## Development Conventions

*   **Code Style:** The code follows the PEP 8 style guide.
*   **Testing:** The project uses `pytest` for testing. Tests are located in the `tests` directory.
*   **Documentation:** The project has extensive documentation in the `docs` directory.
*   **Configuration:** The project configuration is managed in the `config` directory.
*   **Dependencies:** The project's dependencies are listed in the `requirements.txt` file.
