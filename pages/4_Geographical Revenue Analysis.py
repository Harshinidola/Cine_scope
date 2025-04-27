import streamlit as st
import pandas as pd
import plotly.express as px
import ast
import numpy as np

# Language Code to Full Name Mapping
LANGUAGE_MAP = {
    'en': 'English',
    'fr': 'French',
    'es': 'Spanish',
    'de': 'German',
    'ja': 'Japanese',
    'it': 'Italian',
    'ru': 'Russian',
    'hi': 'Hindi',
    'ko': 'Korean',
    'zh': 'Chinese',
    'pt': 'Portuguese',
    'sv': 'Swedish',
    'da': 'Danish',
    'pl': 'Polish',
    'nl': 'Dutch',
    'no': 'Norwegian',
    'fi': 'Finnish',
    'cs': 'Czech',
    'tr': 'Turkish',
    'ar': 'Arabic',
    'he': 'Hebrew',
}

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/movies_metadata.csv", low_memory=False)

    # Keep necessary columns
    df = df[['title', 'revenue', 'production_countries', 'release_date', 'genres', 'original_language']]
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

    # Drop NA and invalid entries
    df = df.dropna(subset=['revenue', 'production_countries', 'release_date', 'genres', 'original_language'])
    df = df[df['revenue'] > 0]

    # Extract release year
    df['release_year'] = df['release_date'].dt.year

    # Parse production countries
    def parse_countries(countries_str):
        try:
            countries = ast.literal_eval(countries_str)
            return [c['name'] for c in countries if isinstance(c, dict) and 'name' in c]
        except:
            return []
    df['production_countries'] = df['production_countries'].apply(parse_countries)

    # Parse genres
    def parse_genres(genres_str):
        try:
            genres = ast.literal_eval(genres_str)
            return [g['name'] for g in genres if isinstance(g, dict) and 'name' in g]
        except:
            return []
    df['genres'] = df['genres'].apply(parse_genres)

    return df

# App title
st.title("🌎 Geographical Revenue Analysis")

# Load data
data = load_data()

# Defaults
min_year = int(data['release_year'].min())
max_year = int(data['release_year'].max())
all_genres = sorted(set(genre for sublist in data['genres'] for genre in sublist))
all_languages = sorted(data['original_language'].dropna().unique())

# Create list for display: full names if available
all_language_full_names = [
    LANGUAGE_MAP.get(lang_code, lang_code) + f" ({lang_code})" for lang_code in all_languages
]
full_name_to_code = {
    LANGUAGE_MAP.get(code, code) + f" ({code})": code for code in all_languages
}

# Expander for filters
with st.expander("🔧 Filter Options", expanded=True):
    year_range = st.slider("Select Year Range", min_year, max_year, (2000, 2010))
    selected_genres = st.multiselect("Select Genre(s)", all_genres, default=all_genres)
    selected_languages_display = st.multiselect("Select Language(s)", all_language_full_names, default=all_language_full_names)

# Convert selected full names back to codes
selected_languages = [full_name_to_code[name] for name in selected_languages_display]

# Apply all filters
filtered_data = data[(
    data['release_year'] >= year_range[0]) & 
    (data['release_year'] <= year_range[1]) & 
    (data['genres'].apply(lambda genres: any(genre in genres for genre in selected_genres))) & 
    (data['original_language'].isin(selected_languages))]

# Prepare data for country revenue
country_revenue = {}

for idx, row in filtered_data.iterrows():
    for country in row['production_countries']:
        if country not in country_revenue:
            country_revenue[country] = 0
        country_revenue[country] += row['revenue']

# Convert to DataFrame
country_rev_df = pd.DataFrame(list(country_revenue.items()), columns=['country', 'revenue'])

# Apply log scale to the revenue to help with color differentiation
country_rev_df['log_revenue'] = np.log1p(country_rev_df['revenue'])

def format_revenue(value):
    if value >= 1_000_000_000:  # Greater than or equal to 1 billion
        return f"${value / 1_000_000_000:,.2f}B"
    elif value >= 1_000_000:  # Greater than or equal to 1 million
        return f"${value / 1_000_000:,.2f}M"
    else:
        return f"${value:,.0f}"  # Display in standard units for smaller values

# Function to format revenue as M or B
def format_revenue(value):
    if value >= 1_000_000_000:  # Greater than or equal to 1 billion
        return f"${value / 1_000_000_000:,.2f}B"
    elif value >= 1_000_000:  # Greater than or equal to 1 million
        return f"${value / 1_000_000:,.2f}M"
    else:
        return f"${value:,.0f}"  # Display in standard units for smaller values

# Choropleth Map
st.subheader(f"🗺️ World Revenue Map - {year_range[0]} to {year_range[1]}")
fig = px.choropleth(
    country_rev_df,
    locations="country",
    locationmode="country names",
    color="log_revenue",
    hover_name="country",
    hover_data={"revenue": ":,.0f"},  # Only show the actual revenue, formatted with commas
    color_continuous_scale="Viridis",
    title=f"Total Revenue by Country ({year_range[0]} - {year_range[1]})",
    labels={'log_revenue': 'Log(Revenue)'},
    height=600
)
fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
st.plotly_chart(fig, use_container_width=True)

# Show total revenue data
st.subheader("📄 Revenue Data by Country")
st.dataframe(country_rev_df.sort_values('revenue', ascending=False))

# 🚀 Top Revenue Movie per Country
st.subheader("🎬 Top Revenue Movie from Each Country")

top_movies = []

for country in country_revenue.keys():
    country_movies = filtered_data[filtered_data['production_countries'].apply(lambda x: country in x)]
    if not country_movies.empty:
        top_movie = country_movies.sort_values('revenue', ascending=False).iloc[0]
        top_movies.append({
            'Country': country,
            'Movie Title': top_movie['title'],
            'Revenue ($)': format_revenue(top_movie['revenue']),
            'Release Year': top_movie['release_year']
        })

top_movies_df = pd.DataFrame(top_movies)
st.dataframe(top_movies_df.sort_values('Revenue ($)', ascending=False))