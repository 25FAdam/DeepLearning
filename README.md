# xView Satellite Imagery Classification

(Deep Learning course - Project) 

<ins>Authors:</ins>
- Ádám Földvári
- Máté Lukács
- Joseph Tartivel

<ins>Instructor:</ins> Roberto Valle

This repository contains the final project for the Deep Learning course at the Universidad Politécnica de Madrid (Master in Digital Innovation - EIT Digital). The task was to classify objects in satellite images using various deep learning techniques, starting from a basic feedforward neural network and progressing through regularization, convolutional networks, and transfer learning.

## Dataset

**Source:** [xView Dataset](https://xviewdataset.org/)
- **Type:** High-resolution satellite images (0.3m GSD, WorldView-3)
- **Split:** 761 training images, 85 testing images
- **Processed:** 21,377 training objects, 2,635 testing objects, cropped and resized to 224x224
- **Classes:** 12 (e.g., building, small car, cargo plane, helicopter, etc.)

## Evaluation Platform

Testing and final evaluation were conducted via [Codabench](https://www.codabench.org/), where we joined a private competition set up by the course instructor. Submissions were made in the required JSON format, and results were benchmarked against the private test set on the platform.

## 📁 Project Structure
```
.
├── ffnn.ipynb                              # Feedforward Neural Network experiments
├── reg.ipynb                               # Regularization techniques applied to ffNNs
├── cnn.ipynb                               # Custom Convolutional Neural Network models
├── tl.ipynb                                # Transfer Learning using pre-trained ResNet50
├── ffnn.png                                # Training loss and accuracy curve for ffNN experiments
├── reg.png                                 # Training loss and accuracy curve for regularized ffNN
├── cnn.png                                 # Training loss and accuracy curve for CNN models
├── tl.png                                  # Training loss and accuracy curve for Transfer Learning model
├── DeepLearning_JT_AF_ML_finalReport.pdf   # Final project report and analysis
└── README.md                               # Project overview, methodology, and results summary
```

Each notebook corresponds to a project phase. We began with a provided template and refined the models across stages.

1. ```ffnn.ipynb``` – **Feedforward Neural Networks**
   - Compared shallow vs. deep ffNNs using flattened image inputs
   - Observed overfitting in deeper models
   - **Best test accuracy:** 45.2%
   - Training loss and accuracy curves for FNN:
   ![ffnn](https://github.com/user-attachments/assets/870bd546-5db4-467d-9c51-da25a9e828c1)
2. ```reg.ipynb``` – **Regularized Feedforward Networks**
   - Added batch normalization, dropout, and increased training time
   - Reduced overfitting, improved generalization
   - **Best test accuracy:** 55.18%
   - Training loss and accuracy curves for regularized FNN:
   ![reg](https://github.com/user-attachments/assets/5d54dbf0-b1ae-4768-a59c-f2ba745791fd)
3. ```cnn.ipynb``` – **Convolutional Neural Networks**
   - Developed CNNs over 5 iterations with improved depth and regularization
   - Included data augmentation and custom learning rate schedules
   - **Best test accuracy:** 76.36%
   - Training loss and accuracy curves for CNN:
   ![cnn](https://github.com/user-attachments/assets/cf67a9a5-157d-4341-8d94-54f43ff2aaff)
4. ```tl.ipynb``` – **Transfer Learning**
   - Fine-tuned pre-trained ResNet50 in two phases:
     - Feature extraction
     - Selective fine-tuning
   - Achieved best results with reduced development time
   - **Best test accuracy:** 77.87%
   - Training loss and accuracy curves for TL:
   ![tl](https://github.com/user-attachments/assets/38878945-2439-4ca6-89b0-88fb2f682aca)

## Results Summary

| Model                  | Test Accuracy | Test Precision | Test Recall |
| ---------------------- | :---: | :---: | :---: |
| **FFNN (Simple)**          | 45.2%         | 30.31%         | 33.66%      |
| **FFNN + Regularization**  | 55.18%        | 44.95%         | 55.5%       |
| **Custom CNN**            | 76.36%        | 74.0%          | 74.53%      |
| **Transfer Learning (TL)** | 77.87%        | 67.15%         | 77.47%      |



## Final Report Summary

This section summarizes our findings and methodology as presented in the full [final report](). The project explored progressively complex deep learning approaches for object classification in satellite imagery.

  ### Phase 1: Feedforward Neural Networks (ffNN)

  We started with basic ffNN architectures using flattened image inputs. Two models were tested: one shallow and one deep.

  - **Challenge:** Loss of spatial structure in flattened data

  - **Result:** Shallow model performed better (**45.2% accuracy**) due to reduced overfitting

  - **Insight:** Complexity does not guarantee improved performance on raw image vectors

  ### Phase 2: Regularized ffNNs

  To improve generalization, we applied:

  - Batch normalization
    
  - Dropout (optimized per layer)
    
  - Increased batch size and training duration
 
  These changes led to a test accuracy of **55.18%**, showing that even non-convolutional models can benefit significantly from regularization strategies.

  ### Phase 3: Convolutional Neural Networks (CNN)

  We developed and iteratively refined five custom CNN models, integrating:

  - Data augmentation (flips, rotations, zoom)

  - L2 regularization

  - Custom learning rate schedules

  - Batch normalization and dropout
 
  The final CNN reached **76.36%** accuracy, substantially outperforming all ffNN variants.

  ### Phase 4: Transfer Learning

  We leveraged a pre-trained ResNet50 model with a two-phase strategy:

  I. Feature Extraction: Train only classification head

  II. Fine-Tuning: Unfreeze top layers and continue training with a lower learning rate

  - Best test accuracy: 77.87%

  - Faster convergence and reduced development time

  - Improved recall on underrepresented classes
 
  ### Key Takeaways
  - **Spatially-aware models** like CNNs and ResNet50 are essential for satellite image analysis
  - **Regularization** significantly improves performance for simpler models
  - **Class imbalance** remains a critical challenge, especially for minority categories like helicopters
  - **Transfer learning** provided the best trade-off between accuracy and development efficiency

  

 ## Infrastructure and Evaluation

- **Platform:** [Kaggle](https://www.kaggle.com/) with P100 GPU acceleration

- **Evaluation:** Conducted via a private competition on [Codabench](https://www.codabench.org/)

- **Submission Format:** JSON files benchmarked against a hidden test set
