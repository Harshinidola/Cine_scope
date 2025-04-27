import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.markdown("---")
st.title("📈 Movie Franchise Performance Over Time")
st.markdown("Analyze how long-running franchises perform across their sequels.")

# Preprocess movies with collections
@st.cache_data
def get_franchise_data():
    movies = pd.read_csv("data/movies_metadata.csv", low_memory=False)
    ratings = pd.read_csv("data/ratings.csv")
    
    # Filter and clean
    movies = movies[['id', 'title', 'release_date', 'belongs_to_collection']]
    movies = movies.dropna(subset=['belongs_to_collection', 'release_date'])

    # Extract collection name
    import ast
    def parse_collection(x):
        try:
            data = ast.literal_eval(x)
            if isinstance(data, dict):
                return data.get('name', None)
        except:
            return None
        return None

    movies['collection'] = movies['belongs_to_collection'].apply(parse_collection)
    movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
    movies = movies.dropna(subset=['collection', 'release_date'])

    # Merge ratings
    ratings['movieId'] = ratings['movieId'].astype(int)
    movies['id'] = pd.to_numeric(movies['id'], errors='coerce').astype('Int64')
    merged = pd.merge(ratings, movies, left_on='movieId', right_on='id', how='inner')

    return merged

franchise_data = get_franchise_data()

# Let user pick a franchise
# Get franchises with movie counts
franchise_counts = franchise_data.groupby('collection')['title'].nunique()
franchise_counts = franchise_counts[franchise_counts > 3].sort_values(ascending=False)

# Create labeled options with counts
franchise_options = [f"{name} ({count} movies)" for name, count in franchise_counts.items()]
franchise_name_map = {f"{name} ({count} movies)": name for name, count in franchise_counts.items()}

selected_label = st.selectbox("Select a movie franchise", franchise_options)
selected_franchise = franchise_name_map[selected_label]

# Filter data
franchise_df = franchise_data[franchise_data['collection'] == selected_franchise]

# Get average rating over time
avg_ratings = franchise_df.groupby(['title', 'release_date']).agg(avg_rating=('rating', 'mean'),
                                                                  count=('rating', 'count')).reset_index()
avg_ratings = avg_ratings.sort_values(by='release_date')

st.markdown(f"### 📊 Ratings Over Time: {selected_franchise}")

# Line chart
fig1 = px.line(avg_ratings, x='release_date', y='avg_rating', text='title',
              markers=True, title=f"{selected_franchise} – Average Ratings Over Time")
fig1.update_traces(textposition="top center")
st.plotly_chart(fig1, use_container_width=True)

# Stacked bar chart of ratings
rating_bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
franchise_df['rating_group'] = pd.cut(franchise_df['rating'], bins=rating_bins, labels=['1★','2★','3★','4★','5★'])
stacked = franchise_df.groupby(['title', 'rating_group']).size().unstack().fillna(0)
stacked = stacked.loc[avg_ratings['title']]  # Keep order

st.markdown(f"### 📊 Rating Distribution Per Movie (Stacked Bars)")

fig2 = px.bar(stacked, title=f"{selected_franchise} – Rating Distribution by Movie", 
              labels={'value': 'Number of Ratings', 'index': 'Movie Title'},
              color_discrete_sequence=px.colors.sequential.RdBu)

st.plotly_chart(fig2, use_container_width=True)
