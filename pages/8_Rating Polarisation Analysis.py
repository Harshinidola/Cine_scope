import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from streamlit_plotly_events import plotly_events
import ast

# Title
st.title('🎬 Polarized Movie Rating Analysis')

# Load data
@st.cache
def load_data():
    ratings = pd.read_csv('data/ratings.csv')
    movies = pd.read_csv('data/movies_metadata.csv', low_memory=False)
    return ratings, movies

ratings, movies = load_data()

# Clean movies
movies = movies[movies['id'].str.isnumeric()].copy()
movies['id'] = movies['id'].astype(int)

if 'genres' in movies.columns:
    movies['genres_list'] = movies['genres'].apply(
        lambda x: [d['name'] for d in ast.literal_eval(x)] if pd.notnull(x) else []
    )
else:
    st.error("Genres column not available in movies_metadata.csv")
    st.stop()

movies.rename(columns={'id': 'movieId'}, inplace=True)
ratings['movieId'] = ratings['movieId'].astype(int)

# Merge ratings + movies
df = ratings.merge(movies[['movieId', 'title', 'release_date', 'poster_path', 'genres_list']], on='movieId')
df['date'] = pd.to_datetime(df['timestamp'], unit='s')

# Extract release year
df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year

# --- 🎯 Select Year ---
st.subheader('🎯 Select Year')

available_years = ['All'] + sorted(df['release_year'].dropna().unique().astype(int).tolist())

selected_year = st.selectbox(
    'Select Year:',
    available_years,
    index=0
)

# Filter by year
if selected_year == 'All':
    df_in_year = df.copy()
else:
    df_in_year = df[df['release_year'] == selected_year]

# Filter movies with at least 5 ratings
rating_counts = df_in_year.groupby('movieId')['rating'].count().reset_index(name='rating_count')
movies_with_enough_ratings = rating_counts[rating_counts['rating_count'] >= 5]['movieId']
df_in_year = df_in_year[df_in_year['movieId'].isin(movies_with_enough_ratings)]

# --- 🎯 Top 5 Polarized Movies ---
if df_in_year.empty:
    st.warning('No movies with enough ratings found for this selection.')
else:
    movie_std = df_in_year.groupby('movieId')['rating'].std().reset_index()
    movie_std = movie_std.merge(df[['movieId', 'title']].drop_duplicates(), on='movieId')
    polarized_std = movie_std.sort_values('rating', ascending=False)
    top5 = polarized_std.head(5)

    st.subheader(f"🥇 Top 5 Most Polarized Movies ({'All Years' if selected_year == 'All' else selected_year})")
    for idx, row in top5.iterrows():
        st.markdown(f"- **{row['title']}** (Std Dev: {row['rating']:.2f})")

    # --- 🎯 Select a Movie Dropdown ---
    st.subheader("🎯 Select a Movie to See Polarization Over Time:")

    top5_movie_names = top5['title'].tolist()

    selected_movie = st.selectbox('Select a Movie:', top5_movie_names)

    if selected_movie:
        selected_movie_id = top5[top5['title'] == selected_movie]['movieId'].values[0]

        st.subheader(f"⏳ Polarization Over Time - {selected_movie}")

        movie_ratings = df[df['movieId'] == selected_movie_id]
        movie_ratings['year_month'] = movie_ratings['date'].dt.to_period('M')

        rating_trend = movie_ratings.groupby('year_month')['rating'].mean().reset_index()
        rating_trend['year_month'] = rating_trend['year_month'].dt.to_timestamp()

        fig3, ax3 = plt.subplots(figsize=(10,6))
        sns.lineplot(x='year_month', y='rating', data=rating_trend, marker='o', ax=ax3)
        ax3.set_xlabel('Year-Month')
        ax3.set_ylabel('Average Rating')
        ax3.grid(True)
        ax3.set_ylim(0,5)
        st.pyplot(fig3)

# --- 📈 Polarization Over Years ---
st.subheader("📈 Average Polarization Over Years")

movie_std_all = df.groupby('movieId')['rating'].std().reset_index()
movie_std_all = movie_std_all.merge(df[['movieId', 'release_year']].drop_duplicates(), on='movieId')
movie_std_all = movie_std_all.dropna(subset=['release_year'])

yearly_polarization = movie_std_all.groupby('release_year')['rating'].mean().reset_index()

fig4, ax4 = plt.subplots(figsize=(12,6))
sns.lineplot(x='release_year', y='rating', data=yearly_polarization, marker='o', color='red', ax=ax4)
ax4.set_xlabel('Release Year')
ax4.set_ylabel('Average Rating Std Dev')
ax4.grid(True)
st.pyplot(fig4)

# --- 📊 Polarization Across Genres ---
st.subheader("📊 Average Polarization by Genre (Click to Explore)")

# Expand genres
df_expanded = df.explode('genres_list')

# Calculate std deviation per movie
movie_std_genre = df_expanded.groupby('movieId')['rating'].std().reset_index()
movie_std_genre = movie_std_genre.merge(df_expanded[['movieId', 'genres_list']].drop_duplicates(), on='movieId')

# Group by genre
genre_polarization = movie_std_genre.groupby('genres_list')['rating'].mean().reset_index()
genre_polarization = genre_polarization.sort_values('rating', ascending=False)

# Plotly colorful bar chart
fig5 = px.bar(
    genre_polarization,
    x='genres_list',
    y='rating',
    color='rating',  # 🎯 Add color
    color_continuous_scale='plasma',  # 🎯 Beautiful gradient
    title='📊 Average Polarization by Genre'
)

fig5.update_layout(
    yaxis=dict(range=[0.8, 1.05]),
    xaxis_title='Genre',
    yaxis_title='Average Rating Standard Deviation',
)

selected_points = plotly_events(fig5, click_event=True, select_event=False)

st.markdown("---")

# --- 📈 After Clicking Genre: Show Movies Barplot ---
if selected_points:
    selected_genre = selected_points[0]['x']
    st.subheader(f"🎯 Polarization Distribution for Movies in Genre: {selected_genre}")

    genre_movies = movie_std_genre[movie_std_genre['genres_list'] == selected_genre]

    if genre_movies.empty:
        st.warning('No movies found for this genre.')
    else:
        # Merge movie names with std dev
        genre_movies = genre_movies.merge(df[['movieId', 'title']].drop_duplicates(), on='movieId')

        # Plot Box Plot
        fig, ax = plt.subplots(figsize=(12,6))
        sns.boxplot(x='title', y='rating', data=genre_movies, ax=ax, palette='viridis')

        ax.set_xlabel('Movies')
        ax.set_ylabel('Rating Std Deviation')
        ax.set_title(f'Polarization (Std Dev) Distribution for Movies in {selected_genre}')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylim(0, 5)  # Limiting Y-axis to rating range
        ax.grid(True)
        st.pyplot(fig)