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
In this part, dataset was preprocessed by removing samples with missing parameters and samples with outliers in its parameters. Outliers were detected through interquartile range where a sample is considered outlier if at least one of its parameter has a value 1.5 times lower/higher than the quartiles 1/3. Shown here the result of preprocessing part:
![image](https://github.com/Rob-Christian/Water-Potability-Prediction/assets/59250293/654390df-fa08-44e8-9b12-c2425738d701)
## Principal Component Analysis
Shown below is the 2D plot of potable vs non-potable water samples:
![image](https://github.com/Rob-Christian/Water-Potability-Prediction/assets/59250293/180ef00d-bb66-47dd-8e85-96bf704ca174)
### Observations
1. Around 26.51% of the total variance were explained by the first two principal components.
2. No clear separation between potable and non-potable samples.
### Implications
1. Feature variables are highly uncorrelated.
2. More principal components are necessary.
3. Dataset is highly nonlinear therefore, a nonlinear model must be used.
## Model Performance
Summarized below the performance of different machine learning algorithms with their best hyperparameter:
![image](https://github.com/Rob-Christian/Water-Potability-Prediction/assets/59250293/510ded20-91d5-4ebd-be7f-9c4012e74cad)
### Observations
1. ANN2 performs best in terms of F1-Score, AUC, TNR, and PPV.
2. SVM and LR performs best in terms of TPR and NPV respectively.
### Implications
1. Balanced complexity since there are only few training variables.
## ANN2 Sensitivity Analysis
Since ANN2 performs best in most of the metrics, neural network sensitivity analysis was applied in the saved architecture for identification of the most important parameter based on perturbation response.
![image](https://github.com/Rob-Christian/Water-Potability-Prediction/assets/59250293/f9a8ce43-e994-4b6c-b934-c8c2f03d423c)
### Observations
1. Hardness input has the highest perturbation response on all inputs
2. All perturbation responses are very low (less than 2%)
### Implications
1. Of all inputs, hardness feature has the greatest impact in terms of water potability
2. The low metrics can be associated to low perturbation response.
# Credits
I would like to give credit to Mr. John Francis Chan for helping me all throughout this study.
