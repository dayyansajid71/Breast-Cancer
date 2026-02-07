# Breast Cancer Prediction Using Machine Learning

## 📌 Overview
This project focuses on the prediction of breast cancer (Benign or Malignant) using machine learning techniques.
The goal is to build a classification model that can assist in early detection of breast cancer based on diagnostic features.

The project is implemented using **Python** and **scikit-learn**, with analysis and visualization performed in **Jupyter Notebooks**.

---

## 📊 Dataset
- The dataset used is provided as `data.csv`
- It contains diagnostic measurements related to breast tumors
- Common features include:
  - Radius
  - Texture
  - Perimeter
  - Area
  - Smoothness, etc.
- Target variable:
  - **Benign**
  - **Malignant**

---

## 🧠 Methodology
The workflow of the project includes:

1. **Data Loading**
   - Importing the dataset using pandas

2. **Data Preprocessing**
   - Handling unnecessary columns
   - Encoding labels
   - Splitting data into training and testing sets
   - Feature scaling (if applicable)

3. **Model Building**
   - Machine learning algorithms such as:
     - Decision Tree Classifier

4. **Model Evaluation**
   - Accuracy score
   - Confusion matrix
   - Model visualization (Decision Tree)

---

## 📈 Results
- The trained model successfully classifies tumors as **Benign** or **Malignant**
- Evaluation metrics such as accuracy and confusion matrix are used to assess performance
- `tree.svg` visualizes the trained decision tree model

---

## 📂 Project Structure
```
Breast-Cancer/
│
├── Breast_Cancer.ipynb
├── Breast_Cancer (1).ipynb
├── data.csv
├── tree.svg
└── README.md
```

---

## ⚙️ Technologies Used
- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib / Seaborn
- Scikit-learn

---

## ▶️ How to Run the Project

1. Clone the repository:
```bash
git clone https://github.com/dayyansajid71/Breast-Cancer.git
```

2. Navigate to the project directory:
```bash
cd Breast-Cancer
```

3. Install required libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

4. Open Jupyter Notebook:
```bash
jupyter notebook
```

5. Run `Breast_Cancer.ipynb`

---

## 🎯 Future Improvements
- Compare multiple ML models (Logistic Regression, SVM, Random Forest)
- Add cross-validation
- Improve feature selection
- Add deep learning approach
- Deploy model using Flask or Streamlit

---

## 👤 Author
**Dayyan Sajid**  
Bioinformatics | Machine Learning | Data Science  

GitHub: https://github.com/dayyansajid71

---

## 📄 License
This project is for educational and research purposes.
