import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
le_gender = LabelEncoder()
le_item = LabelEncoder()
import base64


# Function to convert image to base64
def get_image_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
logo = get_image_base64("logo1.png")
st.sidebar.markdown(f'<img id="telegram_avatar" src="data:image/png;base64,{logo}" style="height:60px; display:inline; margin-right:10px;"/>', unsafe_allow_html=True)

# Upload dataset
st.title("Predict the Top N Probable Items to be Purchased")
st.sidebar.markdown("""
#### Potential benefits of using this approach for business

* **Increased sales:** By predicting the top N probable items to be purchased by a customer, businesses can focus their marketing and sales efforts on those items. This can lead to increased sales of those items and overall revenue for the business.
* **Improved customer satisfaction:** When businesses are able to predict and provide customers with the items they are most likely to want, it can lead to improved customer satisfaction. This is because customers feel like their needs are being anticipated and met.
* **Reduced inventory costs:** By knowing which items are most likely to be purchased, businesses can reduce their inventory costs by only stocking those items. This can free up capital that can be used to invest in other areas of the business.
* **Better decision-making:** By having data on which items are most likely to be purchased, businesses can make better decisions about everything from product development to pricing.

""")
def processData(filename):
    # Label encoding
    df = pd.read_csv(filename)  

    df['Gender'] = le_gender.fit_transform(df['Gender'])
    df['Item Purchased'] = le_item.fit_transform(df['Item Purchased'])
    # Splitting the data
    X = df[['Age', 'Gender']]
    y = df['Item Purchased']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    # Streamlit interface

    # User inputs
    age = st.slider("Select Age", 22, 70)
    gender = st.radio("Select Gender", ["Male", "Female"])
    predicted_items, probabilities, expected_incomes, expected_sales = get_top_n_probable_items(clf, df, age, gender)
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(probabilities, labels=predicted_items, startangle=90, autopct='%1.1f%%')
    ax.axis('equal')
    plt.title('Top N Probable Items')
    st.pyplot(fig)
    # Visualization: Bar Chart for Expected Sales
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.bar(predicted_items, expected_sales, color='lightgreen')
    plt.xlabel('Items')
    plt.ylabel('Expected Sales')
    plt.title('Expected Number of Sales for Top N Probable Items')
    plt.xticks(rotation=45)
    st.pyplot(fig3)
    # Visualization: Bar Chart for Expected Incomes
    print(expected_incomes)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(predicted_items, expected_incomes, color='skyblue')
    plt.xlabel('Items')
    plt.ylabel('Expected Income (USD)')
    plt.title('Expected Income for Top N Probable Items')
    plt.xticks(rotation=45)
    st.pyplot(fig2)
   
def get_top_n_probable_items(clf, df, age, gender, n=10):
    # Encoding the inputs
    gender_encoded = le_gender.transform([gender])[0]
    
    # Predicting probabilities
    probs = clf.predict_proba([[age, gender_encoded]])[0]
    
    # Sorting items based on probabilities
    sorted_indices = probs.argsort()[::-1][:n]
    sorted_probs = probs[sorted_indices]
    sorted_items = le_item.inverse_transform(sorted_indices)
    
    # Calculating expected incomes
    avg_purchase_prices = df.groupby('Item Purchased')['Purchase Amount (USD)'].mean().to_dict()
    avg_prices_sorted = [avg_purchase_prices[le_item.transform([item])[0]] for item in sorted_items]

    # Count of people with inputted age and gender
    customer_count = len(df[(df['Age'] == age) & (df['Gender'] == gender_encoded)])

    # Expected total sales for each item (for all customers with inputted age and gender)
    expected_sales = [round(prob * customer_count) for prob in sorted_probs]

    # Expected total income for each item
    expected_incomes = np.array(avg_prices_sorted) * expected_sales

    return sorted_items, sorted_probs, expected_incomes, expected_sales

# Choice for the user
choice = st.radio('Choose a data source', ['Upload my own dataset', 'Use default dataset'])

if choice == 'Upload my own dataset':
    uploaded_file = st.file_uploader("Upload your dataset (csv file)", type="csv")
    if uploaded_file is not None:
        try:
            processData(uploaded_file)
        except:
            processData('trend.csv')
else:
    processData('trend.csv')