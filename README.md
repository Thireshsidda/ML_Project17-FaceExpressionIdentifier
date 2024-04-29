# ML_Project17-FaceExpressionIdentifier

### Face Expression Recognition with ResNet18
This project implements a Convolutional Neural Network (CNN) for recognizing facial expressions from images. The model is trained on the FER-2013 dataset, which contains images labeled with seven different expressions: neutral, anger, disgust, fear, happiness, sadness, and surprise.

### Getting Started

Requirements: Ensure you have PyTorch, torchvision, and other necessary libraries installed. You can install them using pip install torch torchvision.

Download the Dataset: Download the FER-2013 dataset from Facial Expressions Recognition 2013: https://www.kaggle.com/msambare/fer2013. Extract the downloaded archive and place the folder named 'fer2013' in the same directory as the script.

Run the Script: Execute the Python script (e.g., main.py). The script performs the following steps:

Loads and pre-processes the image data.

Defines a ResNet18 CNN architecture pre-trained on ImageNet, fine-tuned for facial expression classification.

Implements training and validation functions to train the model and evaluate its performance.

Trains the model for a specified number of epochs using a one-cycle learning rate scheduler.

Tracks and displays training metrics (loss, accuracy) during training.

Optionally, plots training and validation curves and learning rate schedule for visualization.


### Code Breakdown:

##### Data Loading and Preprocessing:

The script utilizes ImageFolder from torchvision.datasets to load images from directories.

Data transformations are applied using torchvision.transforms. These transformations include random cropping, horizontal flipping, color jittering (for data augmentation), normalization, and conversion to tensors.

Separate DataLoader instances are created for training, validation, and test datasets (if provided).


##### CNN Model:

The ResNet18 class inherits from a base FacialExpressionClassifier class which defines training, validation, and epoch-end logging functionalities.

The script loads a pre-trained ResNet18 model and replaces the final layer with one having an output size matching the number of expressions (7).

The model uses a combination of convolutional, pooling, and fully-connected layers to extract features and classify facial expressions.


##### Training and Evaluation:

The fit_one_cycle function trains the model for a specified number of epochs using a one-cycle learning rate scheduler. This scheduler dynamically adjusts the learning rate throughout training for better convergence.

The function calculates the training loss and performs validation after each epoch, logging the validation loss and accuracy.
Gradient clipping is optionally applied to prevent exploding gradients.


##### Visualization (Optional):

Functions are provided to plot training and validation accuracy/loss curves and learning rate schedule for better understanding of the training process.

### Further Exploration:

Experiment with different hyperparameters like the number of epochs, learning rate, weight decay, and data augmentation techniques to potentially improve model performance.

Early stopping can be implemented to stop training if validation loss doesn't improve for a certain number of epochs.

The script can be extended to save the trained model for future use on new images.

Explore other CNN architectures like VGG or Inception for facial expression recognition.

Visualize the learned filters in the convolutional layers to understand what features the model focuses on for classification.


### Feel free to modify the code and experiment to enhance the model's performance!
