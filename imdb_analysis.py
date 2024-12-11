from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, lit, desc, split, explode
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("IMDb Movie Analysis") \
    .master("local[*]") \
    .getOrCreate()

# Define schema for movies dataset
movies_schema = StructType([
    StructField("tconst", StringType(), True),
    StructField("titleType", StringType(), True),
    StructField("primaryTitle", StringType(), True),
    StructField("isAdult", IntegerType(), True),
    StructField("startYear", StringType(), True),
    StructField("endYear", StringType(), True),
    StructField("runtimeMinutes", StringType(), True),
    StructField("genres", StringType(), True),
    StructField("averageRating", FloatType(), True),
    StructField("numVotes", IntegerType(), True)
])

# Read the movies data as a stream (from a folder where new files are added)
movies_stream = spark.readStream.format("csv") \
    .option("header", "true") \
    .schema(movies_schema) \
    .load("path_to_streaming_movies_data")

# Define schema for credits dataset
credits_schema = StructType([
    StructField("tconst", StringType(), True),
    StructField("nconst", StringType(), True),
    StructField("category", StringType(), True),
    StructField("job", StringType(), True),
    StructField("characters", StringType(), True)
])

# Load the movies dataset
movies_df = spark.read.csv("path_to_movies_dataset.csv", schema=movies_schema, header=True)

# Load the credits dataset
credits_df = spark.read.csv("path_to_credits_dataset.csv", schema=credits_schema, header=True)

# Filter movies with a minimum of 500 votes
filtered_movies = movies_df.filter(col("numVotes") >= 500)

# Calculate the average number of votes
average_num_votes = filtered_movies.select(avg("numVotes")).first()[0]

# Calculate the ranking score and get the top 10 movies
ranked_movies = filtered_movies.withColumn(
    "rankingScore",
    (col("numVotes") / lit(average_num_votes)) * col("averageRating")
).orderBy(desc("rankingScore")).limit(10)

# List the persons most often credited for the top 10 movies
top_movie_ids = [row["tconst"] for row in ranked_movies.collect()]
top_movies_credits = credits_df.filter(col("tconst").isin(top_movie_ids))
credited_persons = top_movies_credits.groupBy("nconst").agg(count("tconst").alias("count")).orderBy(desc("count"))

# List the different titles of the top 10 movies
top_movie_titles = ranked_movies.select("primaryTitle").distinct()

# Display results
print("Top 10 Movies with Ranking Score:")
ranked_movies.select("primaryTitle", "rankingScore").show()

print("Most Credited Persons for Top 10 Movies:")
credited_persons.show()

print("Different Titles of Top 10 Movies:")
top_movie_titles.show()

# Stop the Spark session
spark.stop()
