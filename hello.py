import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit interface
st.title('Iris Classification with Decision Tree')
st.write(f"Accuracy of the model: {accuracy:.2f}")

# User input for prediction
st.subheader('Make a new prediction')
test_index = st.selectbox('Choose a sample from the test set:', list(range(len(X_test))))

# Display input features and prediction
if st.button('Predict'):
    prediction = clf.predict([X_test[test_index]])
    species = iris.target_names[prediction][0]
    st.write(f'The predicted species is: {species}')
    st.write('Feature values:')
    features = iris.feature_names
    values = X_test[test_index]
    for feature, value in zip(features, values):
        st.write(f"{feature}: {value}")
