#!/usr/bin/env python
# coding: utf-8

# In[39]:


# os libraries
import os 
import itertools 
from PIL import Image

# data handling libraries 
import cv2 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.decomposition import TruncatedSVD
from skimage.transform import resize

#deep learning libraries 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import warnings 
warnings.filterwarnings('ignore')


# In[40]:


def load_mri_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return image

# Path to the training data directory
train_data_directory = 'C:\\Users\\padma\\OneDrive\\Desktop\\Data\\Tumor_Data'

# Lists to store file paths, image data, and labels
image_paths = []
image_data_list = []
category_labels = []

# Iterate through each folder in the training data directory
for folder_name in os.listdir(train_data_directory):
    folder_path = os.path.join(train_data_directory, folder_name)
    
    # Iterate through each file in the folder
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        
        # Load the image data
        img_data = load_mri_image(image_path)
        image_data_list.append(img_data)
        image_paths.append(image_path)
        category_labels.append(folder_name)

# Create pandas Series for file paths, image data, and labels
path_series = pd.Series(image_paths, name='filepaths')
data_series = pd.Series(image_data_list, name='imagedata')
label_series = pd.Series(category_labels, name='labels')

# Combine the Series into a DataFrame
image_data_df = pd.concat([path_series, data_series, label_series], axis=1)

# Display the DataFrame
print(image_data_df.head())


# In[51]:


import random
def plot_random_images(dataframe, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(20, 10))
    
    # Randomly select images
    random_indices = random.sample(range(len(dataframe)), num_images)
    
    for i, ax in enumerate(axes):
        # Access the image data using the 'imagedata' column of the randomly selected index
        img_data = dataframe.iloc[random_indices[i]]['imagedata']
        ax.imshow(img_data, cmap='gray')
        ax.axis('off')
        ax.set_title(f"Label: {dataframe.iloc[random_indices[i]]['labels']}")

    plt.show()

# Call the function with your DataFrame
plot_random_images(image_data_df, num_images=5)


# In[49]:


# Function to resize and then flatten an image
def resize_and_flatten(image, size=(64, 64)):  # Adjust the size as needed
    resized_image = resize(image, size, anti_aliasing=True)
    return resized_image.flatten()

# Apply the function to each image in the DataFrame
image_data_df['flattened_image'] = image_data_df['imagedata'].apply(lambda img: resize_and_flatten(img))

# Convert the list of flattened images to a numpy array
flattened_images = np.vstack(image_data_df['flattened_image'].tolist())

# Perform SVD
n_components = 50  # Adjust as needed
svd = TruncatedSVD(n_components=n_components)
svd.fit(flattened_images)

# Get the transformed data (principal components)
transformed_data = svd.transform(flattened_images)

# Add the SVD features to the DataFrame
for i in range(n_components):
    image_data_df[f'SVD_{i}'] = transformed_data[:, i]

# Print out the SVD features for the first few images
print(image_data_df[[f'SVD_{i}' for i in range(n_components)]].head())


# In[43]:


# Encoding categorical data
label_encoder = LabelEncoder()
image_data_df['encoded_labels'] = label_encoder.fit_transform(image_data_df['labels'])

# Prepare the data for training the classifier
X = image_data_df.filter(regex='SVD_')  # This selects all SVD features
y = image_data_df['encoded_labels']

# Initialize classifiers
classifiers = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='linear', probability=True),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

# Train each classifier and evaluate using cross-validation
for name, classifier in classifiers.items():
    cv_results = cross_validate(classifier, X, y, cv=5,
                                scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'])
    print(f'{name} - CV Accuracy: {np.mean(cv_results["test_accuracy"]):.2f}')
    print(f'{name} - CV Precision Weighted: {np.mean(cv_results["test_precision_weighted"]):.2f}')
    print(f'{name} - CV Recall Weighted: {np.mean(cv_results["test_recall_weighted"]):.2f}')
    print(f'{name} - CV F1 Score Weighted: {np.mean(cv_results["test_f1_weighted"]):.2f}')

# Create a voting classifier
voting_clf = VotingClassifier(estimators=[(name, clf) for name, clf in classifiers.items()], voting='soft')

# Evaluate the voting classifier using cross-validation
cv_results_voting = cross_validate(voting_clf, X, y, cv=5,
                                   scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'])
print('Voting Classifier - CV Accuracy:', np.mean(cv_results_voting['test_accuracy']))
print('Voting Classifier - CV Precision Weighted:', np.mean(cv_results_voting['test_precision_weighted']))
print('Voting Classifier - CV Recall Weighted:', np.mean(cv_results_voting['test_recall_weighted']))
print('Voting Classifier - CV F1 Score Weighted:', np.mean(cv_results_voting['test_f1_weighted']))


# In[44]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train each classifier and evaluate on the test set
for name, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f'{name} - Test Accuracy: {accuracy:.2f}')
    print(f'{name} - Test Precision Weighted: {precision:.2f}')
    print(f'{name} - Test Recall Weighted: {recall:.2f}')
    print(f'{name} - Test F1 Score Weighted: {f1:.2f}\n')

# Train and evaluate the voting classifier
voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)
accuracy_voting = accuracy_score(y_test, y_pred_voting)
precision_voting = precision_score(y_test, y_pred_voting, average='weighted')
recall_voting = recall_score(y_test, y_pred_voting, average='weighted')
f1_voting = f1_score(y_test, y_pred_voting, average='weighted')

print('Voting Classifier - Test Accuracy:', accuracy_voting)
print('Voting Classifier - Test Precision Weighted:', precision_voting)
print('Voting Classifier - Test Recall Weighted:', recall_voting)
print('Voting Classifier - Test F1 Score Weighted:', f1_voting)


# In[65]:


# Train the Random Forest classifier
rf_classifier = classifiers['RandomForest']
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_classifier.predict(X_test)

# Compute the confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)

# Transform encoded labels back to original labels
label_names = label_encoder.inverse_transform(sorted(set(y_test)))

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm_rf, annot=True, fmt='g', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
plt.title('Figure 4: Confusion Matrix for Random Forest Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()



# In[63]:


import pandas as pd
import matplotlib.pyplot as plt
import six

# Data
data = {
    "Classifier": ["Logistic Regression", "Gradient Boosting", "Random Forest", 
                   "SVM", "XGBoost", "Voting Classifier"],
    "Accuracy": [0.79, 0.86, 0.91, 0.81, 0.91, 0.893],
    "F1 Score": [0.78, 0.86, 0.91, 0.81, 0.90, 0.892]
}

# Create DataFrame
df = pd.DataFrame(data)

# Creating a figure to hold the table
fig, ax = plt.subplots(figsize=(8, 3))  # Adjust the size as needed
ax.axis('tight')
ax.axis('off')

# Creating the table and adding it to the figure
the_table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')

# Optional: Auto-size columns
#the_table.auto_set_column_width(col=list(range(len(df.columns))))
plt.suptitle("Figure 3: Classification Model Performance Comparison Table", x=0.5, y=0.05)


plt.show()


# In[74]:


data = {
    "Classifier": ["SVD + Random Forest", "CNN"],
    "Accuracy": [0.91, 0.9618]}
df = pd.DataFrame(data)

# Creating a figure to hold the table
fig, ax = plt.subplots(figsize=(8, 3))  # Adjust the size as needed
ax.axis('tight')
ax.axis('off')

# Creating the table and adding it to the figure
the_table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')

# Optional: Auto-size columns
#the_table.auto_set_column_width(col=list(range(len(df.columns))))
plt.suptitle("Figure 4: Comparison Between my method vs preferred method", x=0.5, y=0.36
            )


plt.show()


# In[ ]:




