# Deep Learning Image Classification Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)
![Status](https://img.shields.io/badge/Status-Learning%20Project-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## About This Project
This is a **learning-focused deep learning project** designed to help understand and implement **Artificial Neural Networks (ANNs)** and **Convolutional Neural Networks (CNNs)** for image classification tasks.

The main purpose is **educational**, to demonstrate how a dataset can be loaded, preprocessed, trained, and evaluated using PyTorch.

---

## Why This Project?
- To practice **deep learning fundamentals**
- To get hands-on experience with **PyTorch**
- To explore **image classification workflows**
- To learn **code structuring for ML projects**

---

## How It Works
1. **Dataset Loading** – Loads images from an open-source dataset (e.g., CIFAR-10).
2. **Preprocessing** – Normalization, resizing, and data augmentation.
3. **Model Training** – CNN model training on GPU/CPU using PyTorch.
4. **Evaluation** – Accuracy and loss plots, prediction samples.
5. **Results Saving** – Saves trained model and metrics.

---

## Project Directory Structure
```plaintext
DeepLearning-ImageClassification/
│
├── datasets/               
├── models/                   
│ ├── ann.py
│ ├── cnn.py                 
├── utils/ 
│ ├── data_loader.py
│ ├── train.py
│ ├── evaluate.py                 
├── main.py                 
├── requirements.txt        
├── README.md               
└── notebooks/              
```

---

## How to Run
```bash
# Clone the repository
git clone https://github.com/AliOding12/deep-learning-image-classification.git

# Navigate into the project directory
cd deep-learning-image-classification

# Install dependencies
pip install -r requirements.txt

# Run the main script
python main.py
```

---

## Example Results
| Epoch | Train Accuracy | Validation Accuracy |
|-------|---------------|---------------------|
| 1     | 65%           | 60%                 |
| 10    | 92%           | 90%                 |

---


## Technologies Used
- **Python 3.8+**
- **PyTorch**
- **Torchvision**
- **Matplotlib**
- **NumPy**

---

## License
This project is licensed under the **MIT License**.

---

> *"The best way to learn Deep Learning is by building projects!"*
<!-- Initial commit: Set up project with README, requirements.txt and .gitignore -->
