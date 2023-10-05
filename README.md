import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create a sample data frame
data = {
    'Age': [55, 19, 50, 21],
    'Gender': ['Male', 'Male', 'Male', 'Male'],
    'Item Purchased': ['Blouse', 'Sweater', 'Jeans', 'Sandals']
}
df = pd.DataFrame(data)

# Convert Gender to numerical values: Male: 1, Female: 0
df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

# Split data into training and test sets
X = df[['Age', 'Gender']]
y = df['Item Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Measure the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Function to predict items for new customers
def predict_item(age, gender):
    gender_num = 1 if gender == 'Male' else 0
    item = clf.predict([[age, gender_num]])[0]
    return item

# Example: Predict item for a 30-year-old female customer
predicted_item = predict_item(30, 'Female')
print(f"A 30-year-old female customer might buy: {predicted_item}")
