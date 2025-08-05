# prodigy_ml_02# Task-02: Customer Segmentation using K-Means Clustering

## 📌 Objective
Create a **K-means clustering algorithm** to group customers of a retail store based on their purchase history.

## 📂 Dataset
The dataset for this project can be found here:  
[Customer Segmentation Tutorial Dataset - Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

## 🛠️ Steps Involved
1. **Import Required Libraries**  
   - pandas, numpy, matplotlib, seaborn, scikit-learn
2. **Load Dataset**  
   - Read the CSV file using `pandas.read_csv()`
3. **Data Preprocessing**  
   - Handle missing values  
   - Select relevant features  
   - Scale the data using `StandardScaler`
4. **Apply K-Means Clustering**  
   - Use the **Elbow Method** to determine the optimal number of clusters  
   - Fit the K-Means model  
   - Assign cluster labels to customers
5. **Visualization**  
   - Use scatter plots and cluster coloring to visualize segmentation
6. **Insights & Conclusion**  
   - Analyze customer groups for targeted marketing

## 📊 Output Example
- Scatter plot of clusters
- Cluster-wise customer segmentation data

## 📎 Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
