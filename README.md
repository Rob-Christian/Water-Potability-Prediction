# Water Potability Prediction
For all human beings, water potability is a must because drinking a non-potable or contaminated water produces illness that might result to death. Here, multiple machine learning algorithms (logistic regression, support vector machines, random forest, and artificial neural networks) were compared for water potability. Shown below is the overall methodology of the study: 
![image](https://github.com/Rob-Christian/Water-Potability-Prediction/assets/59250293/e3d6fdd4-a6df-4314-bc51-c97e00c2db49)
# Dataset
The dataset was obtained from Kaggle, a database used in different researh fields about data science. There are nine (9) input parameters namely; pH value, hardness, total dissolved solids (TDS), chloramines, sulfate, conductivity, total organic carbon (TOC), trihalomethanes, and turbidity. The only output parameter is the water potability. All of the input parameters are continuous while the output parameter is binary (either 0 = non-potable, 1 = potable). Here is the link of the dataset used in the study: https://www.kaggle.com/datasets/adityakadiwal/water-potability
# How to run the project
From the methodology, the codes must be concatenated before running as follows:
1. packages.py
2. functions.py
3. data_preprocessing.py
4. data_visualization_PCA.py
5. data_visualization_TSNE.py
6. logistic_regression.py
7. support_vector_machine.py
8. random_forest.py
9. ann2.py
10. ann4.py
11. ann2_sensitivity&probability_analysis.py
# Results and Discussion
## Data Preprocessing
In this part, dataset was preprocessed by removing samples with missing parameters and samples with outliers in its parameters. Outliers were detected through interquartile range where a sample is considered outlier if at least one of its parameter has a value 1.5 times lower/higher than the quartiles 1/3. Shown here the result of preprocessing part
![image](https://github.com/Rob-Christian/Water-Potability-Prediction/assets/59250293/654390df-fa08-44e8-9b12-c2425738d701)
## Principal Component Analysis
Shown below is the 2D plot of potable vs non-potable water samples. 
![image](https://github.com/Rob-Christian/Water-Potability-Prediction/assets/59250293/180ef00d-bb66-47dd-8e85-96bf704ca174)
### Observations

## Model Performance
## ANN2 Sensitivity Analysis
# Credits
