# streamlit_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import glob

st.set_page_config(layout="wide", page_title="Spotify BigData Dashboard")

# -------------------------
# Config: paths (Spark outputs as directories with part-*.csv files)
# -------------------------
# Helper to get the CSV file from Spark output directory
def get_csv_file(directory):
    files = glob.glob(f"{directory}/part-*.csv")
    return files[0] if files else None

PROCESSED_SONGS = get_csv_file("/Users/amiteshbhaskar/BDA/processed_songs")
CLUSTER_STATS = get_csv_file("/Users/amiteshbhaskar/BDA/cluster_stats")
GENRE_STATS = get_csv_file("/Users/amiteshbhaskar/BDA/genre_stats")
CORRELATIONS = get_csv_file("/Users/amiteshbhaskar/BDA/correlations")

# Helper to load safely
@st.cache_data
def load_csv(path):
    try:
        if path and path.endswith('.csv'):
            # Try with error handling for malformed lines
            return pd.read_csv(path, on_bad_lines='skip', engine='python')
        return None
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")
        return None

# Load data
songs = load_csv(PROCESSED_SONGS)
clusters = load_csv(CLUSTER_STATS)
# yearly = load_csv(YEARLY_TRENDS)  # Not available - no release_year data in this dataset
yearly = None  # Placeholder - no yearly trends data
genres = load_csv(GENRE_STATS)
corrs = load_csv(CORRELATIONS)

# Debug: show columns
if songs is not None:
    st.sidebar.write("📊 Columns:", songs.columns.tolist())
if songs is not None and 'popularity' in songs.columns:
    st.sidebar.write("🎵 Popularity sample:", songs['popularity'].head().tolist())

# Top bar
st.title("🎵 Spotify Big Data — Song Analysis Dashboard")
st.markdown("Explore popularity, mood clusters, feature evolution across years and genres.")

# Layout: left controls, right visuals
left, right = st.columns([1,2])

# Initialize filter variables
ysel = None
artist_sel = []
genre_sel = []

with left:
    st.header("Filters")
    if songs is not None:
        # Year filter only if release_year column exists
        if 'release_year' in songs.columns and songs['release_year'].notna().any():
            years = songs['release_year'].dropna().unique()
            years = sorted([int(y) for y in years if pd.notna(y)])
            if years:
                ysel = st.select_slider("Year range", options=years, value=(years[0], years[-1]))
        
        # Artist filter - check if column exists
        if 'artist' in songs.columns:
            all_artists = songs['artist'].dropna().unique()
            top_artists = songs['artist'].value_counts().index.tolist()[:50]
            artist_sel = st.multiselect("Artists (top 50)", options=top_artists, default=[])
        
        # Genre filter from songs data
        if 'genre' in songs.columns:
            all_genres = songs['genre'].dropna().unique()
            genre_sel = st.multiselect("Genres", options=sorted(all_genres)[:50], default=[])
    
    # Show current filter status
    if artist_sel or genre_sel or ysel:
        st.sidebar.write("**Active Filters:**")
        if ysel:
            st.sidebar.write(f"📅 Years: {ysel[0]}-{ysel[1]}")
        if artist_sel:
            st.sidebar.write(f"🎤 Artists: {len(artist_sel)} selected")
        if genre_sel:
            st.sidebar.write(f"🎵 Genres: {len(genre_sel)} selected")

with right:
    # Apply filters to songs data
    filtered_songs = songs.copy() if songs is not None else None
    if filtered_songs is not None:
        if ysel is not None and 'release_year' in filtered_songs.columns:
            filtered_songs = filtered_songs[(filtered_songs['release_year'] >= ysel[0]) & (filtered_songs['release_year'] <= ysel[1])]
        if artist_sel and 'artist' in filtered_songs.columns:
            filtered_songs = filtered_songs[filtered_songs['artist'].isin(artist_sel)]
        if genre_sel and 'genre' in filtered_songs.columns:
            filtered_songs = filtered_songs[filtered_songs['genre'].isin(genre_sel)]
    
    # Top metrics
    if filtered_songs is not None:
        st.subheader("Top-level metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Songs", f"{len(filtered_songs):,}")
        col2.metric("Unique Artists", f"{filtered_songs['artist'].nunique():,}" if 'artist' in filtered_songs.columns else "N/A")
        
        # Calculate avg popularity more safely
        if 'popularity' in filtered_songs.columns:
            try:
                # Convert to numeric, handling any non-numeric values
                pop_values = pd.to_numeric(filtered_songs['popularity'], errors='coerce')
                avg_pop = pop_values.mean()
                col3.metric("Avg Popularity", f"{avg_pop:.2f}" if not pd.isna(avg_pop) else "N/A")
            except:
                col3.metric("Avg Popularity", "N/A")
        else:
            col3.metric("Avg Popularity", "N/A")

    # Most popular artists (bar)
    if filtered_songs is not None and 'artist' in filtered_songs.columns and 'popularity' in filtered_songs.columns:
        st.subheader("Most Popular Artists")
        try:
            # Ensure popularity is numeric
            filtered_copy = filtered_songs.copy()
            filtered_copy['popularity'] = pd.to_numeric(filtered_copy['popularity'], errors='coerce')
            top_artists = filtered_copy.groupby("artist")['popularity'].mean().reset_index().sort_values("popularity", ascending=False).head(15)
            fig = px.bar(top_artists, x='popularity', y='artist', orientation='h', title="Top Artists by Avg Popularity (Filtered)")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not display artist popularity chart: {e}")

    # Genre stats
    if genres is not None:
        st.subheader("Genres: count vs avg popularity")
        fig = px.scatter(genres, x='count', y='avg_popularity', hover_data=['genre'], title="Genre Popularity vs Count")
        st.plotly_chart(fig, use_container_width=True)

    # Evolution of music features across decades (line)
    if yearly is not None:
        st.subheader("Evolution of Audio Features Over Years")
        # pick a few features to show
        show_features = [c for c in yearly.columns if c.startswith("avg_")]
        # melt for plotting
        yr = yearly.copy()
        yr = yr.sort_values("release_year")
        # Select only key features if too many
        keys = [f for f in show_features if any(k in f for k in ["danceability","energy","valence","tempo","loudness","acousticness"])]
        if len(keys) > 0:
            dfm = yr[['release_year'] + keys].melt(id_vars='release_year', var_name='feature', value_name='avg')
            fig = px.line(dfm, x='release_year', y='avg', color='feature', title="Feature Trends by Year")
            st.plotly_chart(fig, use_container_width=True)

    # Mood clusters visualization (PCA scatter)
    if filtered_songs is not None:
        st.subheader("Mood Clusters (KMeans)")
        # filtered_songs already has filters applied
        if 'pca_x' in filtered_songs.columns and 'pca_y' in filtered_songs.columns:
            hover_cols = []
            for col in ['track', 'artist', 'genre', 'popularity']:
                if col in filtered_songs.columns:
                    hover_cols.append(col)
            fig = px.scatter(filtered_songs.sample(min(2000, len(filtered_songs))), x='pca_x', y='pca_y', color='mood_cluster',
                             hover_data=hover_cols, title=f"Mood Clusters (PCA) - Showing {len(filtered_songs):,} songs")
            st.plotly_chart(fig, use_container_width=True)

    # Correlations
    if corrs is not None:
        st.subheader("Correlation with Popularity")
        st.table(corrs.sort_values('corr_with_popularity', ascending=False).reset_index(drop=True))

st.markdown("---")
st.caption("Data processed by PySpark. Use the filters to refine results.")
