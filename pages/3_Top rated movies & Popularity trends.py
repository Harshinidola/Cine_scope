import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests

st.set_page_config(page_title="Top-Rated Movies & Popularity Trends", layout="wide")

st.title("Top-Rated Movies by Year & Popularity Trends")

TMDB_API_KEY = "1f6516fc1e1c2c873023d5fe5748490d"

@st.cache_data
def load_data():
    metadata = pd.read_csv(r"data\movies_metadata.csv", low_memory=False)
    links = pd.read_csv(r"data\links.csv")

    metadata = metadata[['id', 'title', 'release_date', 'vote_average', 'vote_count', 'genres']]
    metadata['release_date'] = pd.to_datetime(metadata['release_date'], errors='coerce')
    metadata['year'] = metadata['release_date'].dt.year
    metadata['vote_average'] = pd.to_numeric(metadata['vote_average'], errors='coerce')
    metadata['vote_count'] = pd.to_numeric(metadata['vote_count'], errors='coerce')
    metadata = metadata[(metadata['year'] >= 1980)]
    metadata['genres'] = metadata['genres'].fillna("[]").apply(eval).apply(
        lambda g: [d['name'] for d in g] if isinstance(g, list) else []
    )

    links = links[['tmdbId', 'imdbId']].dropna()
    links['tmdbId'] = pd.to_numeric(links['tmdbId'], errors='coerce')
    links['imdbId'] = links['imdbId'].astype(str).str.zfill(7)
    metadata['id'] = pd.to_numeric(metadata['id'], errors='coerce')
    merged = metadata.merge(links, left_on='id', right_on='tmdbId', how='left')

    return merged

df = load_data()

# Sidebar Filters
years = sorted(df['year'].dropna().unique())
years = [y for y in years if 1980 <= y <= 2017]
selected_year = st.sidebar.selectbox("Select Year", years[::-1])

# Added minimum votes filter
min_votes = st.sidebar.slider("Minimum Vote Count", min_value=0, max_value=5000, value=1000, step=10)

all_genres = sorted(set(genre for sublist in df['genres'] for genre in sublist))
selected_genre = st.sidebar.selectbox("Filter by Genre (optional)", ["All"] + all_genres)

# Apply minimum vote count filter
df = df[df['vote_count'] >= min_votes]

# Filtered Data
year_df = df[df['year'] == selected_year]
if selected_genre != "All":
    year_df = year_df[year_df['genres'].apply(lambda g: selected_genre in g)]

top_movies = year_df.sort_values('vote_average', ascending=False).head(5)

# Displaying Top Movies
st.subheader(f"Top 5 Rated Movies of {selected_year}" + (f" (Genre: {selected_genre})" if selected_genre != "All" else ""))
if top_movies.empty:
    st.warning("No movies found with the selected filters.")
else:
    for _, row in top_movies.iterrows():
        col1, col2 = st.columns([1, 4])
        with col1:
            try:
                tmdb_url = f"https://api.themoviedb.org/3/movie/{int(row['id'])}?api_key={TMDB_API_KEY}"
                response = requests.get(tmdb_url)
                if response.status_code == 200:
                    poster_path = response.json().get('poster_path')
                    if poster_path:
                        poster_url = f"https://image.tmdb.org/t/p/w200{poster_path}"
                        st.image(poster_url, width=120)
            except:
                pass
        with col2:
            st.markdown(f"**{row['title']}**")
            st.markdown(f"{row['vote_average']} | {int(row['vote_count'])} votes")
            if pd.notna(row['imdbId']):
                imdb_url = f"https://www.imdb.com/title/tt{row['imdbId']}/"
                st.markdown(f"[IMDb Page]({imdb_url})")

    st.download_button(
        label="Download CSV of Top Movies",
        data=top_movies.to_csv(index=False),
        file_name=f"top_movies_{selected_year}.csv",
        mime="text/csv"
    )

# Popularity Trends
st.subheader("Popularity Trends: Average Rating & #Movies per Year")
summary_df = df.groupby('year').agg({
    'vote_average': 'mean',
    'title': 'count'
}).reset_index().rename(columns={'title': 'number_of_movies'})

fig, ax1 = plt.subplots(figsize=(12, 6))
color = 'tab:blue'
ax1.set_xlabel('Year')
ax1.set_ylabel('Avg Rating', color=color)
ax1.plot(summary_df['year'], summary_df['vote_average'], color=color, marker='o')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('#Movies Rated', color=color)
ax2.bar(summary_df['year'], summary_df['number_of_movies'], alpha=0.3, color=color)
ax2.tick_params(axis='y', labelcolor=color)

st.pyplot(fig)

# TOP-RATED MOVIE EACH YEAR (APPLY GENRE FILTER TOO) 
st.subheader("Top-Rated Movie of Each Year (Comparison View)")

filtered_df = df.copy()
if selected_genre != "All":
    filtered_df = filtered_df[filtered_df['genres'].apply(lambda g: selected_genre in g)]

top_each_year = filtered_df.sort_values(['year', 'vote_average', 'vote_count'], ascending=[True, False, False])
top_movies_per_year = top_each_year.groupby('year').first().reset_index()

fig2, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(top_movies_per_year['year'], top_movies_per_year['vote_average'], color='skyblue')

for bar, title in zip(bars, top_movies_per_year['title']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, title[:15] + '...', 
            ha='center', va='bottom', rotation=90, fontsize=8)

ax.set_xlabel("Year")
ax.set_ylabel("Top Movie Rating")
ax.set_title(f"Top-Rated Movie per Year (since 1980)" + (f" - Genre: {selected_genre}" if selected_genre != "All" else ""))
st.pyplot(fig2)

# Heatmap: Average rating per genre per year
st.subheader("Heatmap: Average Ratings by Genre & Year")

exploded_df = df.explode('genres')
exploded_df = exploded_df.dropna(subset=['genres', 'year'])

genre_year_avg = exploded_df.groupby(['genres', 'year'])['vote_average'].mean().reset_index()
heatmap_data = genre_year_avg.pivot(index='genres', columns='year', values='vote_average')
heatmap_data = heatmap_data.loc[heatmap_data.mean(axis=1).sort_values(ascending=False).index]

fig3, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(heatmap_data, cmap='YlGnBu', linewidths=0.5, linecolor='gray', annot=False, cbar_kws={'label': 'Avg Rating'}, ax=ax)
ax.set_title("Average Rating by Genre and Year", fontsize=14)
ax.set_xlabel("Year")
ax.set_ylabel("Genre")
st.pyplot(fig3)

# Heatmap: Number of movies per genre per year
st.subheader("Heatmap: Number of Movies by Genre & Year")

genre_year_count = exploded_df.groupby(['genres', 'year']).size().reset_index(name='movie_count')
heatmap_count = genre_year_count.pivot(index='genres', columns='year', values='movie_count')
heatmap_count = heatmap_count.loc[heatmap_count.sum(axis=1).sort_values(ascending=False).index]

fig4, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(heatmap_count, cmap='OrRd', linewidths=0.5, linecolor='gray', cbar_kws={'label': '# of Movies'}, ax=ax)
ax.set_title("Number of Movies Released by Genre and Year", fontsize=14)
ax.set_xlabel("Year")
ax.set_ylabel("Genre")
st.pyplot(fig4)

# Footer
st.markdown("---")
st.markdown("This dashboard provides movie insights using MovieLens and TMDb data.")
