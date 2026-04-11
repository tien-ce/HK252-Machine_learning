## **STROKE PREDICTION**

This project applies four machine learning models—Decision Tree, Gradient Boosting, Random Forest, and Support Vector Machine (SVM)—to predict the likelihood of stroke based on patient data.

## System Requirement:
- **Programming Language:** Python 3.10+

## Installation:

- Create virtual environment
```
# Windows
py -m venv venv

# Linux / macOS:
python3 -m venv venv
```

- Activate environment
```
# Windows
venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate
```

- Install libraries from `requirements.txt`
```
pip install -r requirements.txt
```

## Training

- Training Decision_Tree_Model
```
py main.py Decision_Tree_Model
```

- Training Gradient_Boosting_Model
```
py main.py Gradient_Boosting_Model
```

- Training Random_Forest_Library_Model
```
py main.py Random_Forest_Library_Model
```

- Training Random_Forest_Model
```
py main.py Random_Forest_Model
Note: 
It will take quite a long time to train (approximately 20 minutes) since the model uses a basic idea algorithm and does not apply any optimization techniques such as pruning.
```

- Training SVM_Model
```
py main.py SVM_Model
```