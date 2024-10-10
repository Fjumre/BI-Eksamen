import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import plotly.express as px


def load_data():
    df = pd.read_excel('cleaned_marine_microplastics.xlsx')
    return df


df = load_data()

st.title("Plastic and the Oceans")

st.write("""Microplastic pollution in the oceans has become a significant environmental problem, threatening marine life, human health, and the global ecosystem. The spread and concentration of microplastics across different ocean regions, along with their impact on water quality, are issues that require deeper understanding. This project aims to explore patterns in microplastic pollution by analyzing data from various sampling methods, geographical locations, and ocean regions. By identifying high-risk areas, we hope to provide the general public with better insights into the extent of the problem, encouraging more informed actions to reduce pollution and mitigate its effects.""")

st.subheader("Dataset Overview")
st.write(df.head())


# Predicting Ocean Regions
st.write("""This project aimed to explore the spread of microplastic pollution across different oceans and predict pollution levels using features like sampling methods and density ranges. We applied Random Forest models for both regression tasks and classification.""")
st.subheader("Model: Predicting Ocean Regions")

X3 = df[['Sampling Method', 'Measurement', 'Unit', 'Density Class', 'Density Range']]
y3 = df['Oceans']


X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=42)

# Train the Random Forest Classifier for Oceans
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X3_train, y3_train)

# Make predictions and evaluation
y3_pred = rf.predict(X3_test)
accuracy3 = accuracy_score(y3_test, y3_pred)
st.write(f"Accuracy of the Ocean Region model: {accuracy3:.2f}")
st.text("Classification Report for Ocean Regions")
st.write("""The Random Forest Classifier achieved an accuracy of 76% in predicting the ocean based on microplastic measurements, with particularly high precision in the Atlantic ocean. Arctic Ocean: 1, Indian Ocean: 2, Atlantic Ocean: 3, Pacific Ocean: 4""")
st.text(classification_report(y3_test, y3_pred))

# Confusion Matrix for the Ocean Region model
st.write("""The model performed particularly well on Class 3.0 (F1-score: 0.86), which represents the majority ocean region in the dataset. However, the model struggled to accurately classify instances from minority classes, such as Class 1.0 (F1-score: 0.27) and Class 4.0 (F1-score: 0.34).""")
st.subheader("Confusion Matrix for Ocean Region Prediction")
cm = confusion_matrix(y3_test, y3_pred)

fig_cm, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=set(y3), yticklabels=set(y3), ax=ax)
plt.ylabel('Actual Ocean')
plt.xlabel('Predicted Ocean')
plt.title('Confusion Matrix')
st.pyplot(fig_cm)

# Feature Importance for Ocean Regions Model
st.write("""The feature importance plot shows that Measurement is the dominant feature, with an importance score of 0.76. Other features like Unit, Density Range, Sampling Method, and Density Class have much smaller impacts. This suggests that microplastic pollution levels recorded in Measurement drive the predictions, while factors like data collection methods and density class are less influential. Future research should focus on improving the accuracy of Measurement data.""")
st.subheader("Feature Importance for Ocean Region Prediction")
importance3 = rf.feature_importances_
feature_names3 = ['Sampling Method', 'Measurement', 'Unit', 'Density Class', 'Density Range']
fig_feat3, ax = plt.subplots()
sns.barplot(x=importance3, y=feature_names3, ax=ax)
plt.title("Feature Importance in Random Forest (Ocean Regions)")
st.pyplot(fig_feat3)

# Adjust model hyperparameters
st.subheader("Adjust Model Hyperparameters for Ocean Prediction")
n_estimators = st.slider("Number of Trees (n_estimators)", min_value=50, max_value=200, value=100, step=10)
max_depth = st.slider("Maximum Depth of Trees (max_depth)", min_value=5, max_value=50, value=20)

# Rebuild the model based on slider input
rf_updated = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
rf_updated.fit(X3_train, y3_train)
y3_pred_updated = rf_updated.predict(X3_test)

# Display updated classification report
st.write(f"Updated Accuracy with Hyperparameter Adjustment: {accuracy_score(y3_test, y3_pred_updated):.2f}")
st.text("Updated Classification Report for Ocean Regions")
st.text(classification_report(y3_test, y3_pred_updated))

# Model for predicting Density Class
st.subheader("Model: Predicting Density Class")
st.write("""With this model the Density Range feature was the most important predictor of pollution levels, followed by the measurement.
But also a big risk of overfitting. Density Class: Very Low: 1, Low: 2, Medium: 3, High: 4, Very High: 5""")
X4 = df[['Oceans', 'Sampling Method', 'Measurement', 'Unit', 'Density Range']]
y4 = df['Density Class']


# Split the data
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.2, random_state=42)

# Train the Random Forest Classifier for Density Class
clf = RandomForestClassifier()
clf.fit(X4_train, y4_train)

# Make predictions and evaluation
y4_pred = clf.predict(X4_test)
accuracy4 = accuracy_score(y4_test, y4_pred)
st.write(f"Accuracy of the Density Class model: {accuracy4:.2f}")
st.text("Classification Report for Density Class")
st.text(classification_report(y4_test, y4_pred))

# Feature Importance for Density Class Model
st.write("""The Random Forest Classifier relies heavily on Density Range and Measurement for pollution classification, with other features playing smaller roles. These insights can guide future research on microplastic pollution by emphasizing the need for accurate data collection in these key areas. Moreover, the feature importance plot effectively communicates which variables are driving the model's predictions, providing a clear path for future data-driven interventions.""")
st.subheader("Feature Importance for Density Class Prediction")
importance4 = clf.feature_importances_
feature_names4 = ['Oceans', 'Sampling Method', 'Measurement', 'Unit', 'Density Range']
fig_feat4, ax = plt.subplots()
sns.barplot(x=importance4, y=feature_names4, ax=ax)
plt.title("Feature Importance in Random Forest (Density Class)")
st.pyplot(fig_feat4)

# Microplastic Measurement by Ocean Region
st.subheader("Microplastic Measurement by Ocean Region")
st.write("""The bar plot shows that Ocean 4.0 has the highest average pollution, followed by Ocean 3.0, while Ocean 1.0 and Ocean 2.0 have much lower pollution levels. The larger variability in Ocean 3.0 and 4.0 suggests uneven distribution, possibly due to ocean currents or inconsistent sampling. Oceans 1.0 and 2.0 show more consistent, lower pollution levels. This highlights the need to focus mitigation efforts in Oceans 3.0 and 4.0, where the microplastic problem is most severe. Arctic Ocean: 1, Indian Ocean: 2, Atlantic Ocean: 3, Pacific Ocean: 4""")
fig, ax = plt.subplots()
sns.barplot(x='Oceans', y='Measurement', data=df, ax=ax)
plt.title('Microplastic Measurement by Ocean Region')
st.pyplot(fig)
