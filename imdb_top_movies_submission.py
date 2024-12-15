import os
import requests
import gzip
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, lit, desc, broadcast
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

# Function to download and extract IMDb dataset
def download_and_extract_imdb_data(url, output_path, file_name):
    try:
        file_path = os.path.join(output_path, file_name)
        extracted_path = file_path.replace('.gz', '')

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Download file if it doesn't exist
        if not os.path.exists(file_path):
            print(f"Downloading {file_name}...")
            response = requests.get(url, stream=True)
            with open(file_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            print(f"Downloaded {file_name} to {file_path}")

        # Extract file if it hasn't been extracted
        if not os.path.exists(extracted_path):
            print(f"Extracting {file_name}...")
            with gzip.open(file_path, 'rb') as f_in:
                with open(extracted_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Extracted {file_name} to {extracted_path}")

        return extracted_path

    except Exception as e:
        print(f"Error handling file {file_name}: {str(e)}")
        raise

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("IMDb Movie Analysis") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", 8) \
    .config("spark.executor.memory", "2g") \
    .getOrCreate()

# IMDb dataset URLs and paths
output_directory = "imdb_datasets"
ratings_url = "https://datasets.imdbws.com/title.ratings.tsv.gz"
movies_url = "https://datasets.imdbws.com/title.basics.tsv.gz"
principals_url = "https://datasets.imdbws.com/title.principals.tsv.gz"

ratings_file = download_and_extract_imdb_data(ratings_url, output_directory, "title.ratings.tsv.gz")
movies_file = download_and_extract_imdb_data(movies_url, output_directory, "title.basics.tsv.gz")
principals_file = download_and_extract_imdb_data(principals_url, output_directory, "title.principals.tsv.gz")

# Define schemas
ratings_schema = StructType([StructField("tconst", StringType(), True),
                             StructField("averageRating", FloatType(), True),
                             StructField("numVotes", IntegerType(), True)])

movies_schema = StructType([StructField("tconst", StringType(), True),
                            StructField("titleType", StringType(), True),
                            StructField("primaryTitle", StringType(), True),
                            StructField("isAdult", IntegerType(), True),
                            StructField("startYear", StringType(), True),
                            StructField("endYear", StringType(), True),
                            StructField("runtimeMinutes", StringType(), True),
                            StructField("genres", StringType(), True)])

principals_schema = StructType([StructField("tconst", StringType(), True),
                                StructField("ordering", IntegerType(), True),
                                StructField("nconst", StringType(), True),
                                StructField("category", StringType(), True),
                                StructField("job", StringType(), True),
                                StructField("characters", StringType(), True)])

# Load datasets
ratings_df = spark.read.csv(ratings_file, schema=ratings_schema, sep="\t", header=True).cache()
movies_df = spark.read.csv(movies_file, schema=movies_schema, sep="\t", header=True).cache()
principals_df = spark.read.csv(principals_file, schema=principals_schema, sep="\t", header=True).cache()

# Filter movies with a minimum of 500 votes
filtered_ratings = ratings_df.filter(col("numVotes") >= 500)

# Calculate average number of votes
average_num_votes = filtered_ratings.select(avg("numVotes")).first()[0]

# Calculate ranking score and get the top 10 movies
ranked_movies = filtered_ratings.withColumn(
    "rankingScore",
    (col("numVotes") / lit(average_num_votes)) * col("averageRating")
).orderBy(desc("rankingScore"))

# Join with movies dataset and filter by titleType "movie"
top_movies = ranked_movies.join(broadcast(movies_df), "tconst") \
    .filter(col("titleType") == "movie") \
    .select("tconst", "primaryTitle", "rankingScore") \
    .limit(10)

# Get persons most often credited for the top 10 movies by directly joining principals_df
top_movies_credits = principals_df.join(top_movies, "tconst") \
    .groupBy("nconst") \
    .agg(count("tconst").alias("count")) \
    .orderBy(desc("count"))

# Display results in terminal
#ranked_movies_with_titles = ranked_movies.join(broadcast(movies_df), "tconst") \
 #   .filter(col("titleType") == "movie")

# Stop Spark session
spark.stop()
