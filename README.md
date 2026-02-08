# xAI-Proj-B - Group 404

## Requirements
This project requires Python 3 and the following libraries:
- Python 3.x
- Pillow (PIL)
- torch
- torchvision
- scikit-learn
- typing
- numpy
- pandas
- matplotlib

You can install the required packages with:
pip install torch torchvision pillow scikit-learn numpy pandas matplotlib

## Project Structure
The project is organized as follows:

project-root/
│
├── data/                     # Training and testing images
|
├── eval_outputs/          	  # Evaluation results shown in the presentation
│
├── eval_outputs_allPictures/ # Evaluation results shown in the report
│
├── report/                   # LaTeX files for the project report
│
├── results/                  # Training logs and final model weights
|
├── scripts/                  # Notebooks and Python scripts for training and evaluation
│
├── src/                      # Models and dataset classes
│   ├── models/
│   └── datasets/
│
├── .gitattributes
├── LICENSE
└── README.md

## Test image naming conventions
-  `StudentID`
	- FRPK: Elia Maximilian Stamm
	- QJTR: Dominic Christopher Schlegel
	- MZLA: Magdalena Klug

-  `PhoneID`
	- 00001: iPhone 13 Pro
	- 00010: Xiaomi Note 11
	- 00100: Google Pixel 8 Pro
	- 01000: iPhone 16 e