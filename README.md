## Diamond Dynamics – Price Prediction & Market Segmentation
Python Machine Learning Deep Learning Deployment Status

An end-to-end Machine Learning and Deep Learning project to predict diamond prices and segment diamonds into meaningful market categories.
The project is deployed using a Streamlit web application.

## 📌 Project Overview
Diamond pricing depends on several quality attributes such as carat, cut, color, clarity, and dimensions.
This project solves two key business problems:

Predicting diamond prices accurately
Segmenting diamonds into market groups for better pricing and inventory decisions
## 🎯 Objectives
Build multiple regression models for price prediction
Build an Artificial Neural Network (ANN) model
Perform market segmentation using K-Means clustering
Visualize clusters using PCA
Deploy predictions using a Streamlit web app
## 🧠 Skills & Concepts Used
Data Cleaning & Preprocessing
Exploratory Data Analysis (EDA)
Outlier & Skewness Handling
Feature Engineering
Feature Selection
Machine Learning Regression
Artificial Neural Networks (ANN)
K-Means Clustering
PCA (Dimensionality Reduction)
Streamlit Deployment
📊 Dataset Information
Property	Value
Dataset	Diamonds Dataset
Rows	53,940
Columns	10
Main Columns
Column	Description
carat	Weight of the diamond
cut	Cut quality (Fair → Ideal)
color	Color grade (D best → J worst)
clarity	Inclusion quality
x, y, z	Dimensions in mm
price	Price in USD (converted to INR)
🧹 Data Preprocessing
Removed invalid values (0 in x, y, z)
Handled missing values
Converted price from USD to INR
Removed outliers using IQR method
Checked skewness of numerical features
## 📈 Exploratory Data Analysis (EDA)
Price distribution plots
Carat vs price relationship
Price vs cut, color, clarity
Correlation heatmap
Scatter plots and boxplots
## 🧩 Feature Engineering
Derived new features to improve model performance:

Volume = x × y × z
Price per Carat
Dimension Ratio
Carat Category (Light / Medium / Heavy)
## 🎯 Feature Selection
Used Random Forest feature importance
Selected the most influential features for modeling
## 🤖 Regression Models
The following models were trained and evaluated:

Linear Regression
Decision Tree Regressor
Random Forest Regressor
K-Nearest Neighbors (KNN)
XGBoost Regressor (Best Model)
Artificial Neural Network (ANN)
Evaluation Metrics Used

MAE
RMSE
R² Score
## 🧩 Clustering – Market Segmentation
Algorithm: K-Means
Data scaled using StandardScaler
Optimal clusters selected using Elbow Method
PCA used for 2D visualization
Cluster Names
Cluster Name	Description
Affordable Small Diamonds	Low carat, budget-friendly
Mid-range Balanced Diamonds	Medium size and price
Premium Heavy Diamonds	High carat, luxury diamonds
## 🌐 Streamlit Web Application
App Features
Price prediction in INR
Market segment prediction
Preset diamond profiles
Downloadable prediction report
Clean and interactive UI
Run the App
## 📁 Project Structure

diamond-dynamics-ml/
│
├── data/
│   └── diamonds.csv
│
├── models/
│   ├── model.pkl
│   ├── encoder.pkl
│   ├── cluster_model.pkl
│   ├── scaler.pkl
│   └── cluster_names.json
│
├── notebook/
│   └── diamond.ipynb
│
└── diamond.py
## 🚀 Real-World Applications

Dynamic pricing for diamond retailers

Inventory segmentation

Luxury goods recommendation systems

Customer targeting and personalization

## 🛠 Tech Stack

Python

Pandas, NumPy

Scikit-learn

XGBoost

TensorFlow / Keras

Matplotlib, Seaborn

Streamlit

## 👤 Author

Prem Kumar.A

"C:\Users\A Prem kumar\AppData\Local\Programs\Python\Python310\python.exe" -m streamlit run "C:\VSCODE\diamond.py"

