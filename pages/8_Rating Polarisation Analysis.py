import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast

# Title
st.title('Polarized Movie Rating Analysis')

# Load data
@st.cache
def load_data():
    ratings = pd.read_csv('ratings.csv')
    movies = pd.read_csv('movies_metadata.csv', low_memory=False)
    return ratings, movies

ratings, movies = load_data()

# Clean movies data
movies = movies[movies['id'].str.isnumeric()].copy()
movies['id'] = movies['id'].astype(int)

# Check if poster_path available
# has_poster = 'poster_path' in movies.columns

# if has_poster:
#     if 'genres' in movies.columns:
#         movies = movies[['id', 'title', 'release_date', 'poster_path', 'genres']]
#     else:
#         movies = movies[['id', 'title', 'release_date', 'poster_path']]
# else:
#     if 'genres' in movies.columns:
#         movies = movies[['id', 'title', 'release_date', 'genres']]
#     else:
#         movies = movies[['id', 'title', 'release_date']]

movies.rename(columns={'id': 'movieId'}, inplace=True)
ratings['movieId'] = ratings['movieId'].astype(int)

# Extract release year
movies['release_year'] = pd.to_datetime(movies['release_date'], errors='coerce').dt.year

# Merge datasets
df = ratings.merge(movies, on='movieId')

# Convert timestamp to readable date
df['date'] = pd.to_datetime(df['timestamp'], unit='s')

#  Main Page Filter: Select Year
st.subheader('Select Year')

# Sort the available years in descending order
available_years = ['All'] + sorted(df['release_year'].dropna().unique().astype(int).tolist(), reverse=True)

selected_year = st.selectbox(
    'Select Year:',
    available_years,
    index=0  # Default = 'All'
)

# Filter movies released in selected year
if selected_year == 'All':
    df_in_year = df.copy()
else:
    movies_in_year = movies[movies['release_year'] == selected_year]
    df_in_year = df[df['movieId'].isin(movies_in_year['movieId'])]

# Filter movies with at least 5 ratings
rating_counts = df_in_year.groupby('movieId')['rating'].count().reset_index(name='rating_count')
movies_with_enough_ratings = rating_counts[rating_counts['rating_count'] >= 5]['movieId']
df_in_year = df_in_year[df_in_year['movieId'].isin(movies_with_enough_ratings)]

# If no movies after filtering
if df_in_year.empty:
    st.warning('No movies with enough ratings found for this selection.')
else:
    movie_std = df_in_year.groupby('movieId')['rating'].std().reset_index()
    movie_std = movie_std.merge(movies[['movieId', 'title']], on='movieId')
    polarized_std = movie_std.sort_values('rating', ascending=False)
    top5 = polarized_std.head(5)

    # 🌟 Display Top 5 Movie Titles
    st.subheader(f"Top 5 Most Polarized Movies ({'All Years' if selected_year == 'All' else selected_year})")
    for idx, row in top5.iterrows():
        st.markdown(f"- **{row['title']}** (Std Dev: {row['rating']:.2f})")

    # 🌟 Top 5 Polarized Movies Combined Violin Plot
    st.subheader(f"Rating Distribution of Top 5 Polarized Movies")

    top5_movie_ids = top5['movieId'].tolist()
    violin_data = df[df['movieId'].isin(top5_movie_ids)].copy()

    movie_names = dict(zip(movies['movieId'], movies['title']))
    violin_data['movie_name'] = violin_data['movieId'].map(movie_names)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(x='movie_name', y='rating', data=violin_data, inner='quartile', palette='pastel', ax=ax)
    ax.set_xlabel('Movie')
    ax.set_ylabel('Rating')
    ax.set_ylim(0, 5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True)
    st.pyplot(fig)

    # 🌟 Main Page Filter: Select Movie
    st.subheader('Select a Movie to Analyze')
    movie_options = top5['movieId'].tolist()
    selected_movie_id = st.selectbox(
        'Select a Movie:',
        movie_options,
        format_func=lambda x: movies[movies['movieId'] == x]['title'].values[0]
    )

    selected_movie_name = movies[movies['movieId'] == selected_movie_id]['title'].values[0]
    movie_ratings = df[df['movieId'] == selected_movie_id]
    movie_ratings['year_month'] = movie_ratings['date'].dt.to_period('M')

    rating_trend = movie_ratings.groupby('year_month')['rating'].mean().reset_index()
    rating_trend['year_month'] = rating_trend['year_month'].dt.to_timestamp()

    st.subheader(f" Polarization Over Time - {selected_movie_name}")
    fig3, ax3 = plt.subplots(figsize=(10,6))
    sns.lineplot(x='year_month', y='rating', data=rating_trend, marker='o', ax=ax3)
    ax3.set_xlabel('Year-Month')
    ax3.set_ylabel('Average Rating')
    ax3.grid(True)
    ax3.set_ylim(0,5)
    st.pyplot(fig3)

# 📈 Polarization Over Years
st.subheader("Average Polarization Over Years")

movie_std_all = df.groupby('movieId')['rating'].std().reset_index()
movie_std_all = movie_std_all.merge(movies[['movieId', 'release_year']], on='movieId')
movie_std_all = movie_std_all.dropna(subset=['release_year'])

yearly_polarization = movie_std_all.groupby('release_year')['rating'].mean().reset_index()

fig4, ax4 = plt.subplots(figsize=(12,6))
sns.lineplot(x='release_year', y='rating', data=yearly_polarization, marker='o', color='red', ax=ax4)
ax4.set_xlabel('Release Year')
ax4.set_ylabel('Average Rating Std Dev')
ax4.grid(True)
st.pyplot(fig4)

# 📊 Polarization Across Genres (Zoomed Y-axis)
st.subheader("Average Polarization by Genre")

if 'genres' in movies.columns:
    movies['genres_list'] = movies['genres'].apply(lambda x: [d['name'] for d in ast.literal_eval(x)] if pd.notnull(x) else [])
    movie_genres_expanded = movies.explode('genres_list')[['movieId', 'genres_list']]

    movie_std_all_genre = df.groupby('movieId')['rating'].std().reset_index()
    movie_std_all_genre = movie_std_all_genre.merge(movie_genres_expanded, on='movieId')
    movie_std_all_genre = movie_std_all_genre.dropna(subset=['genres_list'])

    genre_polarization = movie_std_all_genre.groupby('genres_list')['rating'].mean().reset_index()
    genre_polarization = genre_polarization.sort_values('rating', ascending=False)

    fig6, ax6 = plt.subplots(figsize=(14,7))
    sns.barplot(x='genres_list', y='rating', data=genre_polarization, palette='viridis', ax=ax6)
    ax6.set_xlabel('Genre')
    ax6.set_ylabel('Average Rating Standard Deviation')
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')
    ax6.set_ylim(0.8, 1.05)
    ax6.grid(True)
    st.pyplot(fig6)
else:
    st.warning('Genres information not available in dataset.')
