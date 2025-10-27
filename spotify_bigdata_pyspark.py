# spotify_bigdata_pyspark_using_your_columns.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, when, split  
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.clustering import KMeans
import pyspark.sql.functions as F
from pyspark.sql.functions import col, when, regexp_extract, lit
import sys
# ---------------------------
# Start Spark session
# ---------------------------
spark = SparkSession.builder \
    .appName("SpotifyBigDataAnalysis") \
    .getOrCreate()

# ---------------------------
# CONFIG — change if needed
# ---------------------------
INPUT_CSV = "/Users/amiteshbhaskar/BDA/SpotifyFeatures.csv"   # or local "spotify_songs.csv"
OUTPUT_DIR = "/Users/amiteshbhaskar/BDA" # where to save outputs (can be local path)
NUM_CLUSTERS = 6
SEED = 42

# ---------------------------
# Load CSV (with header & inferSchema)
# ---------------------------
df = spark.read.option("header", True).option("inferSchema", True).csv(INPUT_CSV)
print("Initial row count:", df.count())

# ---------------------------
# Rename common columns (ensure consistency)
# ---------------------------
# Your columns: genre, artist_name, track_name, track_id, popularity,
# acousticness, danceability, duration_ms, energy, instrumentalness, key, liveness,
# loudness, mode, speechiness, tempo, time_signature, valence

# For convenience, create shorter aliases used later:
if 'artist_name' in df.columns:
    df = df.withColumnRenamed('artist_name', 'artist')
if 'track_name' in df.columns:
    df = df.withColumnRenamed('track_name', 'track')
if 'track_id' in df.columns:
    df = df.withColumnRenamed('track_id', 'track_id')  # keep as-is

# ---------------------------
# Cast feature columns to double
# ---------------------------
if "time_signature" in df.columns:
    df = df.withColumn(
        "time_signature_num",
        F.when(
            col("time_signature").rlike(r"^\d+/\d+$"),
            F.regexp_extract(col("time_signature"), r"^(\d+)/\d+$", 1)
        ).otherwise(col("time_signature"))
    )
    df = df.drop("time_signature").withColumnRenamed("time_signature_num", "time_signature")

feature_cols = ["acousticness", "danceability", "duration_ms", "energy",
                "instrumentalness", "key", "liveness", "loudness", "mode",
                "speechiness", "tempo", "valence", "popularity", "time_signature"]

for c in feature_cols:
    if c in df.columns:
        # Use SQL try_cast which returns NULL on invalid input instead of erroring
        df = df.withColumn(c, F.expr(f"try_cast({c} as double)"))

# ---------------------------
# Release year handling (optional)
# ---------------------------
# If your CSV has 'release_date' or 'year' use it; otherwise yearly trends will be skipped.
# if 'release_date' in df.columns:
#     df = df.withColumn("release_year",
#                        when(col("release_date").rlike(r"^\d{4}$"), col("release_date").cast("int"))
#                        .otherwise(year(F.to_date("release_date", "yyyy-MM-dd"))))
# elif 'year' in df.columns:
#     df = df.withColumn("release_year", col("year").cast("int"))
# else:
#     df = df.withColumn("release_year", F.lit(None).cast("int"))
#     print("No 'release_date' or 'year' found — yearly trends will be skipped.")

# ---------------------------
# Drop rows with nulls in core features
# ---------------------------
core_features = ["danceability","energy","valence","tempo","popularity"]
existing_core = [c for c in core_features if c in df.columns]
df = df.na.drop(subset=existing_core)
print("Row count after dropping nulls in core features:", df.count())

# ---------------------------
# Correlation calculation (feature vs popularity)
# ---------------------------
corrs = []
corr_features = ["danceability","energy","loudness","speechiness","acousticness",
                 "instrumentalness","liveness","valence","tempo","duration_ms"]
for feature in corr_features:
    if feature in df.columns:
        c_val = df.stat.corr(feature, "popularity")
        corrs.append((feature, float(c_val) if c_val is not None else None))

corr_df = spark.createDataFrame(corrs, ["feature","corr_with_popularity"])
corr_df.coalesce(1).write.mode("overwrite").option("header",True).csv(f"{OUTPUT_DIR}/correlations")
print("Saved correlations to", f"{OUTPUT_DIR}/correlations")

# ---------------------------
# Clustering (KMeans) — choose features appropriate for mood
# ---------------------------
cluster_features = ["danceability","energy","valence","tempo","loudness","acousticness","instrumentalness","liveness"]
available_cluster_features = [c for c in cluster_features if c in df.columns]
if len(available_cluster_features) < 3:
    print("Not enough features for clustering. Available:", available_cluster_features)
    # Save minimal outputs and exit gracefully
    corr_df.coalesce(1).write.mode("overwrite").option("header",True).csv(f"{OUTPUT_DIR}/correlations")
    spark.stop()
    sys.exit(0)

assembler = VectorAssembler(inputCols=available_cluster_features, outputCol="features_raw")
df = assembler.transform(df)

scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
scaler_model = scaler.fit(df)
df = scaler_model.transform(df)

kmeans = KMeans(k=NUM_CLUSTERS, seed=SEED, featuresCol="features", predictionCol="mood_cluster")
kmodel = kmeans.fit(df)
df = kmodel.transform(df)

print("KMeans training complete. Cluster centers (scaled):")
for i, center in enumerate(kmodel.clusterCenters()):
    print(f"Cluster {i}: {center}")

# ---------------------------
# PCA -> 2D for visualization
# ---------------------------
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.ml.linalg import SparseVector, DenseVector

def get_pca_value(vector, index):
    try:
        if isinstance(vector, DenseVector):
            return float(vector.values[index])
        elif isinstance(vector, SparseVector):
            if index in vector.indices:
                return float(vector.values[list(vector.indices).index(index)])
            return 0.0
    except:
        return None

pca = PCA(k=2, inputCol="features", outputCol="pca_features")
pca_model = pca.fit(df)
df = pca_model.transform(df)

get_pca_udf = udf(lambda v, i: get_pca_value(v, i), DoubleType())
df = df.withColumn("pca_x", get_pca_udf(col("pca_features"), F.lit(0)))
df = df.withColumn("pca_y", get_pca_udf(col("pca_features"), F.lit(1)))

# ---------------------------
# Cluster stats: counts & avg features
# ---------------------------
agg_exprs = [F.count("*").alias("count")]
for f in available_cluster_features:
    agg_exprs.append(F.avg(F.col(f)).alias(f"avg_{f}"))

cluster_stats = df.groupBy("mood_cluster").agg(*agg_exprs).orderBy("mood_cluster")
cluster_stats.coalesce(1).write.mode("overwrite").option("header",True).csv(f"{OUTPUT_DIR}/cluster_stats")
print("Saved cluster stats to", f"{OUTPUT_DIR}/cluster_stats")

# ---------------------------
# Save processed songs (select columns)
# ---------------------------
out_cols = ["track","artist","genre","track_id","popularity","mood_cluster","pca_x","pca_y"] + available_cluster_features
out_cols = [c for c in out_cols if c in df.columns]
df.select(*out_cols).coalesce(1).write.mode("overwrite").option("header",True).csv(f"{OUTPUT_DIR}/processed_songs")
print("Saved processed songs to", f"{OUTPUT_DIR}/processed_songs")
# ...existing code...
# ---------------------------

# ---------------------------
# Genre stats
# ---------------------------
if 'genre' in df.columns:
    genre_stats = df.groupBy("genre").agg(F.count("*").alias("count"), F.avg("popularity").alias("avg_popularity")) \
                    .orderBy(F.desc("count"))
    genre_stats.coalesce(1).write.mode("overwrite").option("header",True).csv(f"{OUTPUT_DIR}/genre_stats")
    print("Saved genre stats to", f"{OUTPUT_DIR}/genre_stats")

# ---------------------------
# Yearly trends (only if release_year present)
# ---------------------------
# if df.filter(col("release_year").isNotNull()).count() > 0:
#     yearly = df.groupBy("release_year").agg(*[F.avg(c).alias(f"avg_{c}") for c in available_cluster_features + ["popularity"]]) \
#                .orderBy("release_year")
#     yearly.coalesce(1).write.mode("overwrite").option("header",True).csv(f"{OUTPUT_DIR}/yearly_trends")
#     print("Saved yearly trends to", f"{OUTPUT_DIR}/yearly_trends")
# else:
#     print("No release_year data — skipping yearly trends.")

# ---------------------------
# Stop Spark
# ---------------------------
spark.stop()
print("PySpark job complete.")
