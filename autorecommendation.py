import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset (Modify with your own data)
data = {
    "Item": [
        "Laptop", "Smartphone", "Tablet", "Smartwatch", "Headphones", "Camera", "Gaming Console", "Speaker", "Monitor", "Keyboard"
    ],
    "Description": [
        "A powerful laptop with high-speed performance and great battery life.",
        "A smartphone with an excellent camera and smooth user experience.",
        "A portable tablet perfect for reading, browsing, and light work.",
        "A smartwatch with health tracking and notifications on the go.",
        "Wireless noise-canceling headphones with immersive sound.",
        "A high-resolution camera with amazing photo quality.",
        "A gaming console with 4K graphics and an extensive game library.",
        "A Bluetooth speaker with deep bass and long battery life.",
        "A high-refresh-rate monitor for gaming and professional work.",
        "A mechanical keyboard with RGB lighting and fast response."
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Vectorize the descriptions
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Description'])

# Compute Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Streamlit App
st.title("Auto Recommendation System")
st.write("Select an item to get recommendations.")

# User selection
selected_item = st.selectbox("Choose an item:", df["Item"].tolist())

if st.button("Get Recommendations"):
    idx = df[df["Item"] == selected_item].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:4]  # Get top 3 recommendations
    recommended_items = [df.iloc[i[0]]["Item"] for i in sim_scores]
    
    st.subheader("Recommended Items:")
    for item in recommended_items:
        st.write(f"- {item}")
