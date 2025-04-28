import streamlit as st
import pandas as pd
import ast
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
import plotly.graph_objects as go

st.set_page_config(page_title="Genre Evolution Over Time", layout="wide")
st.title("Genre Evolution Over Time")

@st.cache_data
def load_data():
    df = pd.read_csv('data/movies_metadata.csv', low_memory=False)
    df = df.dropna(subset=['release_date', 'genres'])
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df = df.dropna(subset=['release_date'])
    df['year'] = df['release_date'].dt.year
    def parse_genres(x):
        if pd.isnull(x) or x == '[]':
            return []
        try:
            parsed = ast.literal_eval(x)
            return [genre['name'] for genre in parsed]
        except:
            return []
    df['genres'] = df['genres'].apply(parse_genres)
    return df

df = load_data()
df_exploded = df.explode('genres')
unique_genres = sorted(df_exploded['genres'].dropna().unique().tolist())

st.subheader("Number of Movies Released per Genre Each Year")
with st.expander("Filters for Bar Chart", expanded=True):
    bar_genres = st.multiselect("Select Genres (Bar Chart)", options=unique_genres, default=unique_genres, key="bar_genres")
    bar_year_range = st.slider("Select Year Range (Bar Chart)", int(df_exploded['year'].min()), 2017, (1980, 2017), key="bar_year")

bar_filtered_df = df_exploded[
    (df_exploded['genres'].isin(bar_genres)) &
    (df_exploded['year'] >= bar_year_range[0]) &
    (df_exploded['year'] <= bar_year_range[1])
]

bar_genre_year_counts = bar_filtered_df.groupby(['year', 'genres']).size().reset_index(name='count')

fig_bar = px.bar(
    bar_genre_year_counts,
    x="year",
    y="count",
    color="genres",
    hover_data={'year': True, 'genres': True, 'count': True},
    labels={"year": "Year", "count": "Number of Movies", "genres": "Genre"},
    title="Movies Released per Genre Each Year",
)
fig_bar.update_layout(
    barmode='stack',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color="white"),
    xaxis=dict(showgrid=False, linecolor='gray', tickfont=dict(color='white')),
    yaxis=dict(showgrid=False, linecolor='gray', tickfont=dict(color='white')),
    legend_title=dict(font=dict(color="white")),
    legend=dict(font=dict(color="white")),
    title=dict(x=0.5, xanchor='center', font=dict(color="white"))
)
st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("Genre Popularity Trends Over Time")
with st.expander("Filters for Line Chart", expanded=True):
    line_genres = st.multiselect("Select Genres (Line Chart)", options=unique_genres, default=unique_genres, key="line_genres")
    line_year_range = st.slider("Select Year Range (Line Chart)", int(df_exploded['year'].min()), 2017, (1980, 2017), key="line_year")

line_filtered_df = df_exploded[
    (df_exploded['genres'].isin(line_genres)) &
    (df_exploded['year'] >= line_year_range[0]) &
    (df_exploded['year'] <= line_year_range[1])
]

line_genre_year_counts = line_filtered_df.groupby(['year', 'genres']).size().reset_index(name='count')
line_pivot = line_genre_year_counts.pivot(index='year', columns='genres', values='count').fillna(0)

fig_line, ax_line = plt.subplots(figsize=(15, 7))
palette = sns.color_palette("tab10", len(line_genres))
for idx, genre in enumerate(line_genres):
    if genre in line_pivot.columns:
        ax_line.plot(line_pivot.index, line_pivot[genre], label=genre, color=palette[idx % len(palette)], linewidth=2)

ax_line.set_xlabel("Year", fontsize=12, color='white')
ax_line.set_ylabel("Number of Movies", fontsize=12, color='white')
ax_line.set_title("Genre Popularity Over Time", fontsize=16, color='white')
ax_line.legend(title='Genres', bbox_to_anchor=(1.05, 1), loc='upper left')
sns.despine()
fig_line.patch.set_alpha(0.0)
ax_line.set_facecolor('none')
for spine in ['bottom', 'left']:
    ax_line.spines[spine].set_color('gray')
    ax_line.spines[spine].set_linewidth(1.2)
ax_line.spines['top'].set_visible(False)
ax_line.spines['right'].set_visible(False)
ax_line.tick_params(axis='x', colors='white')
ax_line.tick_params(axis='y', colors='white')

st.pyplot(fig_line)

st.subheader("🎥 Animated Genre Popularity Over Time")

with st.expander("Filters for Animated Chart", expanded=True):
    anim_genres = st.multiselect(
        "Select Genres", 
        options=unique_genres, 
        default=unique_genres, 
        key="anim_genres"
    )
    anim_year_range = st.slider(
        "Select Year Range", 
        int(df_exploded['year'].min()), 
        2017, 
        (1980, 2017), 
        key="anim_year"
    )

anim_filtered_df = df_exploded[
    (df_exploded['genres'].isin(anim_genres)) &
    (df_exploded['year'] >= anim_year_range[0]) &
    (df_exploded['year'] <= anim_year_range[1])
]

anim_genre_year_counts = anim_filtered_df.groupby(['year', 'genres']).size().reset_index(name='count')

fig_animated = px.bar(
    anim_genre_year_counts,
    x="genres",
    y="count",
    animation_frame="year",
    color="genres",
    range_y=[0, anim_genre_year_counts['count'].max() + 20] if not anim_genre_year_counts.empty else [0, 1],
    labels={"genres": "Genre", "count": "Number of Movies"},
    title="Genre Popularity Animation"
)

fig_animated.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color="dimgray"),
    xaxis=dict(showgrid=False, linecolor='gray', tickfont=dict(color='dimgray')),
    yaxis=dict(showgrid=False, linecolor='gray', tickfont=dict(color='dimgray')),
)

st.plotly_chart(fig_animated, use_container_width=True)



st.subheader("Full Interactive Genre Co-occurrence Network")
with st.expander("Filters for Network Graph", expanded=True):
    net_genres = st.multiselect("Select Genres (Network)", options=unique_genres, default=unique_genres, key="net_genres")
    net_year_range = st.slider("Select Year Range (Network)", int(df_exploded['year'].min()), 2017, (1980, 2017), key="net_year")

co_occurrence = {}
for genres in df[df['year'].between(net_year_range[0], net_year_range[1])]['genres']:
    genres = [g for g in genres if g in net_genres]
    for i in range(len(genres)):
        for j in range(i + 1, len(genres)):
            key = tuple(sorted((genres[i], genres[j])))
            co_occurrence[key] = co_occurrence.get(key, 0) + 1

G = nx.Graph()
genre_counts = {}
for genres in df[df['year'].between(net_year_range[0], net_year_range[1])]['genres']:
    for g in genres:
        if g in net_genres:
            genre_counts[g] = genre_counts.get(g, 0) + 1
for genre, count in genre_counts.items():
    G.add_node(genre, size=count)
for (g1, g2), count in co_occurrence.items():
    G.add_edge(g1, g2, weight=count)

pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)

edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines'
)

node_x = []
node_y = []
node_size = []
node_text = []

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_size.append(G.nodes[node]['size'] / 50 + 10)
    node_text.append(f"{node}<br>{G.nodes[node]['size']} movies")

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=[node for node in G.nodes()],
    textposition="bottom center",
    hovertext=node_text,
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        color=[G.nodes[node]['size'] for node in G.nodes()],
        size=node_size,
        colorbar=dict(
            thickness=15,
            title=dict(text='Number of Movies', side='right'),
            xanchor='left'
        ),
        line_width=2
    )
)

fig_network = go.Figure(data=[edge_trace, node_trace],
    layout=go.Layout(
        title='<br>Genre Co-occurrence Network',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[dict(
            text="Drag nodes to explore connections!",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002)],
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        paper_bgcolor='black',
        plot_bgcolor='black'
))

st.plotly_chart(fig_network, use_container_width=True)
