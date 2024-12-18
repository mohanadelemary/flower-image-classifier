# Flower Images Classifier Project

This project develops a CLI app with an training and prediction ML pipeline to detect and classify flower images. The app trains on pre-labeled dataset. The app itself serves as an input interface to analyze and classify new flower images. The program is coded to adopt a wide range of input pre-trained models. 


### Instructions:
1. Clone the repository.

2. Install the required dependencies by running the following command in the project's root directory:
        `pip install -r requirements.txt`

3. Run the following commands in the project's root directory to set up your database and model.

    - Train
        Train a new network on a data set with `train.py`

        * Basic usage: `python train.py data_directory`
        * Prints out training loss, validation loss, and validation accuracy as the network trains
        * Options: * Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory` * Choose architecture: `python train.py data_dir --arch "vgg13"` * Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20` * Use GPU for training: `python train.py data_dir --gpu`

    - Predict
        Predict flower name from an image with `predict.py` along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

        * Basic usage: `python predict.py /path/to/image checkpoint`
        * Options: * Return top K most likely classes: `python predict.py input checkpoint --top_k 3` * Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json` * Use GPU for inference: `python predict.py input checkpoint --gpu`


## Repository Structure

The project repository is organized as follows:

```bash
flower-image-classifier/  
│
├── cat_to_name.json                   # JSON file with dictionary to map classes to flowers names
├── checkpoint.pth                     # Trained model saved from running the ipynb
├── flower_data.tar.gz                 # Compressed directory with flowers images
├── Image_Classifier_Project.ipynb     # Jupyter Notebook with initial code and testing
├── README.md                          # ReadMe file with project documentation
├── requirements.txt                   # File with all project requirements
├── data_prep.py                       # Python script to process user input and network build up
├── predict.py                         # Python script to predict flower species of an input image
└── train.py                           # Python script to train and save a custom pre-traing NN
