# Bayesian Neural Networks for Pulsar Candidate Identification

## Description
This project implements Bayesian Convolutional Neural Networks for pulsar candidate identification, addressing distribution discrepancies through multimodal incremental learning. The implementation is based on PyTorch and focuses on handling evolving RFI environments.

## Key Features
- Bayesian CNN implementation with variational inference
- Multimodal data processing for pulsar diagnostic plots
- Incremental learning capability
- Distribution shift handling
- Uncertainty estimation

## Project Structure
```python
.
├── loss.py                         # Loss function implementations
├── main.py                         # Main entry point
├── multimodel.py                   # Multimodal model architecture
├── multimodel_four.py             # Four-input multimodal implementation
├── multimodel_fourBayesian.py     # Bayesian version of four-input model
├── training.py                    # Training utilities
├── training_CL.py                 # Continual learning training
├── training_four.py              # Four-input model training
└── training_four_Bayesian.py     # Bayesian model training


## Software and Tools

This project implements Bayesian Convolutional Neural Networks using PyTorch. Below are the main tools and libraries that have been utilized:

### Main Dependencies
- **PyTorch**: An open-source machine learning library used for applications such as computer vision and natural language processing, primarily developed by Facebook's AI Research lab (FAIR).
- **NumPy**: A library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

### Installation
To install PyTorch and other necessary libraries, run the following command:
```bash
pip install torch torchvision numpy
```


## References

This implementation is based on the following academic papers:

- Kumar Shridhar, Felix Laumann, Marcus Liwicki. "A comprehensive guide to Bayesian Convolutional Neural Network with variational inference." *arXiv preprint arXiv:1901.02731* (2019).

- Kumar Shridhar, Felix Laumann, Marcus Liwicki. "Uncertainty estimations by softplus normalization in Bayesian Convolutional Neural Networks with variational inference." *arXiv preprint arXiv:1806.05978* (2018).

## License

Distributed under the MIT License. See LICENSE for more information.
