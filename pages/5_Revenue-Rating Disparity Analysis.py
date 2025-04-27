import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import ast
import requests

# TMDB API Key
TMDB_API_KEY = "1f6516fc1e1c2c873023d5fe5748490d"

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/movies_metadata.csv", low_memory=False)

    df = df[['title', 'revenue', 'vote_average', 'vote_count', 'release_date', 'genres']]
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
    df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')
    df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce')
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

    df = df.dropna(subset=['revenue', 'vote_average', 'vote_count', 'release_date'])
    df = df[(df['revenue'] > 0) & (df['vote_average'] > 0) & (df['vote_count'] >= 10)]

    # df['norm_revenue'] = (df['revenue'] - df['revenue'].min()) / (df['revenue'].max() - df['revenue'].min())
    # df['norm_rating'] = (df['vote_average'] - df['vote_average'].min()) / (df['vote_average'].max() - df['vote_average'].min())
    # df['disparity'] = df['norm_revenue'] - df['norm_rating']

    df['z_revenue'] = (df['revenue'] - df['revenue'].mean()) / df['revenue'].std()
    df['z_rating'] = (df['vote_average'] - df['vote_average'].mean()) / df['vote_average'].std()
    df['disparity'] = df['z_revenue'] - df['z_rating']


    df['release_year'] = df['release_date'].dt.year

    def parse_genres(genre_str):
        try:
            genres = ast.literal_eval(genre_str)
            return [g['name'] for g in genres if isinstance(g, dict) and 'name' in g]
        except:
            return []
    df['genres'] = df['genres'].apply(parse_genres)

    return df

# Fetch poster
@st.cache_data(show_spinner=False)
def fetch_poster(title):
    try:
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
        response = requests.get(search_url)
        if response.status_code == 200:
            results = response.json().get('results')
            if results:
                poster_path = results[0].get('poster_path')
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w200{poster_path}"
    except:
        pass
    return None

# Streamlit Styling
st.markdown("""
    <style>
        body {
            background-color: #f9f9f9;
        }
        .stApp {
            font-family: 'Segoe UI', sans-serif;
        }
        div[data-testid="stExpander"] > summary {
            font-weight: 600;
            font-size: 1.05rem;
        }
    </style>
""", unsafe_allow_html=True)

# App title
st.title("Revenue-Rating Disparity Analysis")

# Load data
data = load_data()

# Defaults
min_year = int(data['release_year'].min())
max_year = int(data['release_year'].max())
all_genres = sorted(set(genre for sublist in data['genres'] for genre in sublist))

# Expander for filters
with st.expander("🔧 Filter Options", expanded=True): 
    st.markdown("**📅 Release Year Range**")
    year_range = st.slider("", min_year, max_year, (1998, 2007), key="year_slider")

    st.markdown("**🎭 Filter by Genre(s)**")
    select_all_genres = st.checkbox("Select All Genres", value=True, key="select_all_genres")
    if select_all_genres:
        selected_genres = st.multiselect("", all_genres, default=all_genres, key="genre_multiselect")
    else:
        selected_genres = st.multiselect("", all_genres, key="genre_multiselect")

    st.markdown("**🗳️ Minimum Vote Count**")
    min_votes = st.number_input("", min_value=0, value=1000, step=10, key="min_votes_input")

# Apply filters
filtered_data = data[
    (data['vote_count'] >= min_votes) &
    (data['release_year'] >= year_range[0]) &
    (data['release_year'] <= year_range[1]) &
    (data['genres'].apply(lambda x: any(g in x for g in selected_genres)))
]

# Scatter plot
st.markdown("---")
st.subheader("📈 Scatter Plot: Ratings vs Revenue")
fig1 = px.scatter(
    filtered_data, x='vote_average', y='revenue', hover_name='title',
    size='vote_count', color='disparity',
    color_continuous_scale='RdBu',
    #color_continuous_midpoint = 0,
    color_continuous_midpoint=filtered_data['disparity'].median(),
    labels={'vote_average': 'Rating', 'revenue': 'Revenue'}
)
fig1.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(size=14)
)
st.plotly_chart(fig1)

# Movie cards with poster and genres
st.markdown("---")
def display_movie_cards(df, title):
    st.markdown(f"### {title}")
    for _, row in df.iterrows():
        col1, col2 = st.columns([1, 4])
        with col1:
            poster_url = fetch_poster(row['title'])
            if poster_url:
                st.image(poster_url, width=150)
        with col2:
            st.markdown(f"**{row['title']}**")
            #st.markdown(f"📅 **Year:** {row['release_year']}")
            genre_text = ", ".join(row['genres']) if row['genres'] else "N/A"
            #st.markdown(f"🎭 *Genres:* {genre_text}")
            st.markdown(
                f"📅 **Year:** {row['release_year']}  \n" 
                f"🎭 **Genres:** {genre_text}  \n"
                f"⭐ **Rating:** {row['vote_average']}  \n"
                f"💰 **Revenue:** ${int(row['revenue']):,}  \n"
                f"📈 **Disparity:** {row['disparity']:.2f}"
            )

display_movie_cards(filtered_data.sort_values('disparity', ascending=False).head(5), "🔝 Top 5 Positive Disparity (High revenue compared to rating)")
display_movie_cards(filtered_data.sort_values('disparity', ascending=True).head(5), "🔻 Top 5 Negative Disparity (Low revenue compared to rating)")
