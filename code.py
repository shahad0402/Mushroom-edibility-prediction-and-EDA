#  It starts by importing the necessary libraries for data analysis and visualization: NumPy, Pandas, Matplotlib, and Seaborn.
#
# The next line of code reads a CSV file called "Mushroom Edibility.csv" into a Pandas DataFrame named "df". The "sep" parameter is used to indicate that the columns in the CSV file are separated by semicolons instead of commas.
#
# Finally, the code outputs the contents of the DataFrame "df" to the Colab console. This allows the user to examine the data and ensure that it was imported correctly before proceeding with any further analysis.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('/content/Mushroom Edibility.csv',sep=';')
df

# Adding df.head() to the code would print the first five rows of the DataFrame "df". This would allow the user to get a quick overview of the data and see what the columns and values look like.

df.head()

df['cap-shape']

# Adding df.tail() to the code would print the last five rows of the DataFrame "df". This would allow the user to see if there are any trends or patterns in the data at the end of the file that are not visible at the beginning.

df.tail()

# Adding df.columns to the code would print a list of all the column names in the DataFrame "df". This can be helpful for checking the column names and making sure they match what is expected.

df.columns

# Adding df.describe() to the code would generate a statistical summary of the DataFrame "df". This summary includes the count, mean, standard deviation, minimum, and maximum values for each numerical column in the DataFrame.

df.describe()

# Adding df.info() to the code would print a concise summary of the DataFrame "df". This includes information about the total number of rows, the number of non-null values in each column, and the data type of each column.

df.info()

# Adding df.dtypes to the code would print the data type of each column in the DataFrame "df". This can be helpful for checking if the data types are correct and consistent across all columns.

df.dtypes

# Adding df.shape to the code would return a tuple that shows the number of rows and columns in the DataFrame "df". The first value in the tuple represents the number of rows and the second value represents the number of columns.

df.shape

# Adding df['class'].nunique() to the code would return the number of unique values in the "class" column of the DataFrame "df". In this case, since the "class" column represents whether each mushroom is edible or poisonous, the output would be 2.

df['class'].nunique()

# Adding df['class'].value_counts() to the code would return a count of unique values in the "class" column of the DataFrame "df". In this case, since the "class" column represents whether each mushroom is edible or poisonous, the output would show the count of edible and poisonous mushrooms in the dataset.

df['class'].value_counts()

# Adding df.isna().sum() to the code would show the count of missing values in each column of the DataFrame "df". This can be helpful for identifying any missing data that needs to be dealt with before analysis.

df.isna().sum()

# This code imports the LabelEncoder class from the sklearn.preprocessing module and initializes an instance of it called le.
#
# The code then loops through each column in the DataFrame "df". If the data type of the column is an object (indicated by 'O'), the code uses the fit_transform() method of the LabelEncoder instance to transform the values in that column to numeric labels.
#
# This is often done when working with machine learning models, as many models require the input data to be numeric. By using LabelEncoder, we can convert categorical variables to numeric labels so that they can be used in the models.
#
# The resulting DataFrame "df" will have the same columns and rows as the original DataFrame, but with any object columns converted to numeric labels.

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

for col in df.columns:
    if df[col].dtypes=='O':
        df[col]=le.fit_transform(df[col])

# df
#
# Adding df.hist(figsize=(15,10)) to the code would create a histogram for each numerical column in the DataFrame "df". The figsize parameter sets the size of the resulting figure.
#
# Histograms are a useful way to visualize the distribution of values in a dataset, and can help to identify any patterns or outliers.
#
# The resulting output would show a grid of histograms, one for each numerical column in the DataFrame. Each histogram would show the frequency of values in that column across a range of values.

df.hist(figsize=(15,10))

df.isna().sum()

# This code creates a heatmap using the Seaborn library to visualize the correlation between variables in the DataFrame "df".
#
# The plt.figure(figsize=(25,10)) sets the size of the resulting figure.
#
# The code computes the correlation between all pairs of variables in the DataFrame using the corr() method. The resulting correlation matrix is stored in a variable called cor.
#
# The Seaborn library is then used to create a heatmap of the correlation matrix, with each cell in the heatmap color-coded to indicate the strength and direction of the correlation between the two variables it represents. The annot=True parameter shows the correlation coefficients in each cell of the heatmap.
#
# The resulting output would show a heatmap with a color scale that ranges from dark red (indicating a strong positive correlation) to white (indicating no correlation) to dark blue (indicating a strong negative correlation). The diagonal of the heatmap would be all red, indicating the correlation of each variable with itself.

plt.figure(figsize=(25,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

# This code creates a boxplot using the Seaborn library to visualize the distribution of values for three numerical columns in the DataFrame "df": 'cap-diameter', 'stem-height', and 'stem-width'.
#
# The f, ax = plt.subplots(figsize=(10, 8)) creates a new figure with a specified size.
#
# The Seaborn library is then used to create a boxplot of the three columns, with each column represented by a box-and-whisker plot. The box represents the interquartile range (IQR) of the data, with the whiskers extending to the minimum and maximum values that are within 1.5 times the IQR of the lower and upper quartiles. Any points beyond the whiskers are plotted as outliers.
#
# The resulting output would show a boxplot with three boxes, one for each column, and the x-axis would show the name of each column. The y-axis would show the range of values for each column. Any outliers would be plotted as individual points beyond the whiskers.

f, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df[['cap-diameter', 'stem-height', 'stem-width']])

sns.boxplot(x=df['class'], y=df['cap-diameter'], palette='deep')
print()

sns.boxplot(x=df['class'], y=df['stem-width'], palette='deep')
print()

sns.boxplot(x=df['class'], y=df['stem-height'], palette='deep')
print()

# df['cap-color'].value_counts()
#
# This code creates a countplot using the Seaborn library to visualize the frequency of each value in the 'cap-color' column of the DataFrame "df", with the hue separated by the 'class' column.
#
# The sns.set_palette(sns.color_palette(['red','black'])) sets the color palette to red and black.
#
# The f, ax = plt.subplots(figsize=(8, 5)) creates a new figure with a specified size.
#
# The Seaborn library is then used to create a countplot of the 'cap-color' column, with the frequency of each value shown on the y-axis and the different values of the 'class' column separated by hue. This plot shows the number of edible and poisonous mushrooms for each color of the cap.

sns.set_palette(sns.color_palette(['red','black']))
f, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x=df['cap-color'], hue=df['class'])
print()

sns.set_palette(sns.color_palette(["red", "black"]))
f, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x=df['veil-color'], hue=df['class'])
print()

sns.set_palette(sns.color_palette(["red", "black"]))
f, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x=df['spore-print-color'], hue=df['class'])
print()

sns.set_palette(sns.color_palette(["red", "black"]))
f, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x=df['stem-surface'], hue=df['class'])
print()

sns.set_palette(sns.color_palette(["red", "black"]))
f, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x=df['stem-root'], hue=df['class'])
print()

sns.set_palette(sns.color_palette(["red", "black"]))
f, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x=df['habitat'], hue=df['class'])
print()
#
# This code creates a violin plot using the Seaborn library to visualize the distribution of 'stem-height' values for each value in the 'veil-color' column of the DataFrame "df". The violin plot is separated by the 'class' column using hue.
#
# The sns.set_palette(sns.color_palette(["red", "black"])) sets the color palette to red and black.
#
# The f, ax = plt.subplots(figsize=(8, 5)) creates a new figure with a specified size.
#
# The Seaborn library is then used to create a violin plot of the 'stem-height' column, with each violin representing the distribution of values for a different value of the 'veil-color' column. The 'class' column is separated by hue. The violin plot shows the distribution of values for each value of the 'veil-color' column by plotting a kernel density estimate on either side of a box plot, which represents the quartiles and median of the data.

sns.set_palette(sns.color_palette(["red", "black"]))
f, ax = plt.subplots(figsize=(8, 5))
sns.violinplot(x='veil-color', y='stem-height', data=df, hue='class')
print()

sns.set_palette(sns.color_palette(["red", "black"]))
f, ax = plt.subplots(figsize=(8, 5))
sns.violinplot(x='veil-color', y='stem-width', data=df, hue='class')
print()
#
# This code extracts the feature variables and target variable from the DataFrame "df".
#
# The df.iloc[:,1:] selects all rows and all columns starting from the second column (index 1) to the end. This selects all the feature variables in the DataFrame.
#
# The .values attribute returns a Numpy array containing the values in the selected DataFrame.
#
# The resulting array x contains all the feature variables in the DataFrame.
#
# The df['class'] selects the 'class' column in the DataFrame.
#
# The .values attribute returns a Numpy array containing the values in the selected DataFrame.
#
# The resulting array y contains the target variable in the DataFrame.

x=df.iloc[:,1:].values
y=df['class'].values
#
# This code uses the train_test_split function from the sklearn.model_selection module to split the feature variables (x) and target variable (y) into training and testing sets.

The train_test_split(x, y, test_size=0.30, random_state=42) function takes four arguments:

# x: the feature variables array
# y: the target variable array
# test_size=0.30: the proportion of the data to use for testing (in this case, 30% of the data is used for testing)
# random_state=42: the random seed used for the random sampling of the data. This ensures that the same split is obtained each time the code is run.
# The function returns four arrays:
# #
# x_train: the feature variables array for the training set
# x_test: the feature variables array for the testing set
# y_train: the target variable array for the training set
# y_test: the target variable array for the testing set
# The resulting output splits the data into training and testing sets, with 70% of the data used for training and 30% of the data used for testing.

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)

# This code imports all functions and classes from the sklearn.metrics module.
#
# The metrics module in scikit-learn contains a variety of functions for computing various classification and regression metrics. These metrics are used to evaluate the performance of machine learning models on test data.
#
# Some examples of the functions available in the sklearn.metrics module are accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, etc.

from sklearn.metrics import*

# This code trains a decision tree classifier model using the training data (x_train and y_train).

# The DecisionTreeClassifier class from the sklearn.tree module is used to create a decision tree classifier object (D_model), which is trained using the .fit() method with the training data.
#
# The criterion parameter is set to 'entropy', which means that the information gain is calculated using the entropy measure.
#
# The trained model is used to make predictions on the testing set (x_test) using the .predict() method, and the predicted labels are stored in y_pred.
#
# The accuracy_score() function from the sklearn.metrics module is used to compute the accuracy of the predicted labels (y_pred) with respect to the true labels (y_test).
#
# The classification_report() function is used to print a text report showing the main classification metrics (precision, recall, f1-score, and support) for each class.
#
# The confusion_matrix() function is used to compute the confusion matrix for the predicted and true labels.
#
# The ConfusionMatrixDisplay() class from the sklearn.metrics module is used to create a confusion matrix plot. The result and cmd variables are passed as parameters to the ConfusionMatrixDisplay() constructor to display the confusion matrix and the class labels, respectively. The resulting confusion matrix plot is displayed using the .plot() method.

from sklearn.tree import DecisionTreeClassifier
D_model=DecisionTreeClassifier(criterion='entropy')
D_model.fit(x_train,y_train)
y_pred=D_model.predict(x_test)
print("score is:",accuracy_score(y_pred,y_test))
print("*"*100)
print(classification_report(y_test,y_pred))
print("*"*100)
result=confusion_matrix(y_test,y_pred)
print(result)
print("*"*100)
cmd=['edible','poisunos']
cm=ConfusionMatrixDisplay(result,display_labels=cmd)
cm.plot()


#This code trains a k-nearest neighbors classifier model using the training data (x_train and y_train).

#The KNeighborsClassifier class from the sklearn.neighbors module is used to create a k-nearest neighbors classifier object (K_model), which is trained using the .fit() method with the training data.

#The n_neighbors parameter is set to 5, which means that the classifier uses the 5 nearest neighbors to predict the label of a new instance.

#The trained model is used to make predictions on the testing set (x_test) using the .predict() method, and the predicted labels are stored in y_pred.

#The accuracy_score() function from the sklearn.metrics module is used to compute the accuracy of the predicted labels (y_pred) with respect to the true labels (y_test).

#The classification_report() function is used to print a text report showing the main classification metrics (precision, recall, f1-score, and support) for each class.

#The confusion_matrix() function is used to compute the confusion matrix for the predicted and true labels.

#The ConfusionMatrixDisplay() class from the sklearn.metrics module is used to create a confusion matrix plot. The k_result and cmd variables are passed as parameters to the ConfusionMatrixDisplay() constructor to display the confusion matrix and the class labels, respectively. The resulting confusion matrix plot is displayed using the .plot() method.

from sklearn.neighbors import KNeighborsClassifier
K_model=KNeighborsClassifier(n_neighbors=5)
K_model.fit(x_train,y_train)
y_pred=K_model.predict(x_test)
print("score is:",accuracy_score(y_pred,y_test))
print("*"*100)
print(classification_report(y_test,y_pred))
print("*"*100)
k_result=confusion_matrix(y_test,y_pred)
print(k_result)
print("*"*100)
cmd=['edible','poisunos']
cm=ConfusionMatrixDisplay(k_result,display_labels=cmd)
cm.plot()

#It seems that the code is performing classification on a dataset containing information about mushroom edibility. It uses various libraries such as numpy, pandas, matplotlib, and seaborn for data manipulation, visualization, and classification purposes. The dataset is first loaded into a pandas dataframe and then preprocessed by encoding categorical variables using label encoding. Then, the data is split into training and testing sets, and three classification models (Decision Tree, K-Nearest Neighbors, and Support Vector Machines) are applied to the training data. The performance of each model is evaluated by computing various metrics such as accuracy, classification report, and confusion matrix, and visualized using a confusion matrix display.

from sklearn.svm import SVC
s_model=SVC()
s_model.fit(x_train,y_train)
y_pred=s_model.predict(x_test)
print("score is:",accuracy_score(y_pred,y_test))
print("*"*100)
print(classification_report(y_test,y_pred))
print("*"*100)
s_result=confusion_matrix(y_test,y_pred)
print(s_result)
print("*"*100)
cmd=['edible','poisunos']
cm=ConfusionMatrixDisplay(s_result,display_labels=cmd)
cm.plot()