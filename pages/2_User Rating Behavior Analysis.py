import streamlit as st
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans
import plotly.express as px

st.title("🎭 User Rating Pattern Analysis")

# Load data from local file
try:
    uploaded_file = 'data/ratings_small.csv'
    ratings = pd.read_csv(uploaded_file)
    required_cols = {'userId', 'movieId', 'rating', 'timestamp'}
    if not required_cols.issubset(ratings.columns):
        missing = required_cols - set(ratings.columns)
        st.error(f"Missing columns: {', '.join(missing)}")
        st.stop()
except Exception as e:
    st.error(f"Error reading file: {str(e)}")
    st.stop()

# Calculate user statistics with variance
user_stats = ratings.groupby('userId').agg(
    rating_variance=('rating', 'var'),
    avg_rating=('rating', 'mean'),
    rating_frequency=('movieId', 'count')
).reset_index()
user_stats['rating_variance'] = user_stats['rating_variance'].fillna(0)

# 3D visualization with variance
st.subheader("Rating Analysis Plot")
fig_raw = px.scatter_3d(
    user_stats, 
    x='rating_variance', 
    y='avg_rating', 
    z='rating_frequency',
    color='avg_rating',
    size='rating_frequency',
    hover_data=['userId', 'rating_variance', 'avg_rating', 'rating_frequency'],
    color_continuous_scale='rainbow',
    opacity=0.7,
    title='User Rating Pattern Analysis: Variance vs Average vs Frequency'
)
fig_raw.update_layout(
    scene=dict(
        xaxis_title='Rating Variance',
        yaxis_title='Average Rating',
        zaxis_title='Rating Frequency'
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    coloraxis_colorbar=dict(title="Avg Rating"),
    scene_camera=dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=-0.2),
        eye=dict(x=1.5, y=1.5, z=0.4)
    )
)
st.plotly_chart(fig_raw, use_container_width=True)

st.subheader("🌐 Overall Rating Histogram")

# Plot histogram of all ratings
fig_hist = px.histogram(
    ratings, 
    x='rating', 
    nbins=20, 
    title='Overall Rating Histogram',
    labels={'rating': 'Rating'},
    color_discrete_sequence=['#FF4B4B']
)

fig_hist.update_layout(
    template="plotly_dark",
    xaxis_title="Rating",
    yaxis_title="Frequency",
    showlegend=False
)

st.plotly_chart(fig_hist, use_container_width=True)



# GMM feature extraction with 3 components
def extract_gmm_features(user_ratings):
    if len(user_ratings) < 3:  # Minimum 3 ratings for 3-component GMM
        return None
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    gmm.fit(user_ratings.reshape(-1,1))
    return np.concatenate([
        gmm.weights_,
        gmm.means_.flatten(),
        gmm.covariances_.flatten()
    ])

# Process users with progress bar
users = []
intermediate_features = []
with st.spinner("Analyzing user rating patterns..."):
    for user_id, group in ratings.groupby('userId'):
        features = extract_gmm_features(group['rating'].values)
        if features is not None:
            users.append(user_id)
            intermediate_features.append({
                'UserID': user_id,
                'Weight1': features[0],
                'Weight2': features[1],
                'Weight3': features[2],
                'Mean1': features[3],
                'Mean2': features[4],
                'Mean3': features[5],
                'Cov1': features[6],
                'Cov2': features[7],
                'Cov3': features[8],
                'RatingCount': len(group)
            })
    if not intermediate_features:
        st.error("No valid users found with sufficient ratings")
        st.stop()

# Create DataFrames
intermediate_df = pd.DataFrame(intermediate_features)
feature_cols = [
    'Weight1', 'Weight2', 'Weight3',
    'Mean1', 'Mean2', 'Mean3',
    'Cov1', 'Cov2', 'Cov3',
    'RatingCount'
]
X = intermediate_df[feature_cols].values

# Kernel transformation pipeline
with st.spinner("Transforming features..."):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
    X_transformed = kpca.fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_transformed)

# Visualization
cluster_df = pd.DataFrame({
    'UserID': users,
    'Cluster': clusters,
    'Component1': X_transformed[:,0],
    'Component2': X_transformed[:,1],
    'RatingCount': intermediate_df['RatingCount']
})

st.subheader("Cluster Analysis Results of GMM Features")
fig_cluster = px.scatter(
    cluster_df, 
    x='Component1', 
    y='Component2',
    color='Cluster',
    color_discrete_sequence=px.colors.qualitative.Vivid,
    size='RatingCount',
    hover_data=['UserID'],
    title="User Clusters in Kernel PCA Space"
)
st.plotly_chart(fig_cluster, use_container_width=True)


# Cluster analysis
st.subheader("📊 Cluster Characteristics")
cluster_stats = cluster_df.groupby('Cluster').agg(
    UserCount=('UserID', 'count'),
    Avg_RatingCount=('RatingCount', 'mean'),
    Component1_Mean=('Component1', 'mean'),
    Component2_Mean=('Component2', 'mean')
).reset_index()
st.dataframe(cluster_stats.style.format("{:.2f}"), use_container_width=True)

# Cluster distribution reconstruction
st.subheader("📈 Cluster Rating Distributions")

# Merge cluster assignments with GMM parameters
cluster_params = pd.merge(
    cluster_df[['UserID', 'Cluster']], 
    intermediate_df,
    on='UserID'
)

# Calculate mean GMM parameters per cluster
cluster_centers = cluster_params.groupby('Cluster').agg({
    'Weight1': 'mean',
    'Weight2': 'mean',
    'Weight3': 'mean',
    'Mean1': 'mean',
    'Mean2': 'mean',
    'Mean3': 'mean',
    'Cov1': 'mean',
    'Cov2': 'mean',
    'Cov3': 'mean'
}).reset_index()

# Prepare data for distribution plot
x_vals = np.linspace(0.5, 5, 500)
plot_data = []
for _, row in cluster_centers.iterrows():
    cluster_num = row['Cluster']
    weights = [row['Weight1'], row['Weight2'], row['Weight3']]
    means = [row['Mean1'], row['Mean2'], row['Mean3']]
    covs = [row['Cov1'], row['Cov2'], row['Cov3']]
    
    pdf = np.zeros_like(x_vals)
    for w, m, c in zip(weights, means, covs):
        pdf += w * (1/np.sqrt(2*np.pi*c)) * np.exp(-0.5*((x_vals - m)**2)/c)
    pdf /= np.trapezoid(pdf, x_vals)  # Normalize
    
    freq = cluster_stats[cluster_stats['Cluster'] == cluster_num]['Avg_RatingCount'].values[0]
    
    for x, y in zip(x_vals, pdf):
        plot_data.append({
            'Cluster': f'Cluster {cluster_num}',
            'Rating': x,
            'Density': y,
            'Avg Frequency': freq
        })

plot_df = pd.DataFrame(plot_data)

# Create visualization with frequency encoding
fig = px.line(
    plot_df,
    x='Rating',
    y='Density',
    color='Cluster',
    line_group='Cluster',
    hover_name='Cluster',
    hover_data={'Avg Frequency': ':.1f'},
    title='Cluster Rating Distributions '
)

# Scale line widths based on frequency
min_freq = plot_df['Avg Frequency'].min()
max_freq = plot_df['Avg Frequency'].max()
line_widths = np.interp(plot_df['Avg Frequency'].unique(),
                        [min_freq, max_freq],
                        [1, 4])

for idx, cluster in enumerate(plot_df['Cluster'].unique()):
    fig.data[idx].line.width = line_widths[idx]

fig.update_layout(
    annotations=[
        dict(
            text=f"Line thickness ↔ Cluster frequency",
            xref="paper", yref="paper",
            x=0.95, y=0.85,
            showarrow=False,
            font=dict(size=10)
        )
    ]
)

# Display
col1, col2 = st.columns([2, 1])
with col1:
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
---
## 📝 **Cluster Reviews and Behavioral Insights**

### **Cluster 0.0: The Discerning Enthusiasts**
- **Distribution:** Extremely sharp, tall peak near rating 4.5–5.
- **Behavior:**  
  - These users rate most movies very highly, with little variation.
  - Likely *enthusiastic fans* or only rate movies they already expect to enjoy.
  - Consistent high ratings suggest a positive bias or selective reviewing.
  - **As critics:** Not harsh; reviews are overwhelmingly positive, possibly less helpful for finding flaws.

---

### **Cluster 1.0: The Balanced Critics**
- **Distribution:** Broad, bell-shaped curve centered around 3.5.
- **Behavior:**  
  - Use the full rating scale, with a moderate average.
  - More variance, likely to give both low and high ratings.
  - Balanced, neither too harsh nor too lenient.
  - **As critics:** Fair and nuanced, offering a realistic sense of a movie’s quality.

---

### **Cluster 2.0: The Polarized Raters**
- **Distribution:** Bimodal, with peaks around 2 and 4.
- **Behavior:**  
  - Tend to love or hate movies, rarely giving middle ratings.
  - Decisive and possibly more emotional in their judgments.
  - Reviews may be more extreme, useful for identifying divisive films.
  - **As critics:** Passionate, valuable for spotting strong reactions, less so for nuanced takes.

---

### **Cluster 3.0: The Positive Discerners**
- **Distribution:** Peak around 4, with moderate spread.
- **Behavior:**  
  - Generally positive but with more diversity than Cluster 0.
  - Willing to rate lower than 5 but still skew high.
  - Appreciate quality but not afraid to mark down when warranted.
  - **As critics:** Optimistic but not blindly so, offering praise with some critical discernment.
""")