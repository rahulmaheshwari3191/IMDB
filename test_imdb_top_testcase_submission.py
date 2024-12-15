import unittest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, avg, desc, count
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
import os
import requests
import gzip
import shutil

# Helper function to download and extract IMDb datasets
def download_and_extract_imdb_data(url, output_path, file_name):
    file_path = os.path.join(output_path, file_name)
    extracted_path = file_path.replace('.gz', '')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(file_path):
        response = requests.get(url, stream=True)
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)

    if not os.path.exists(extracted_path):
        with gzip.open(file_path, 'rb') as f_in:
            with open(extracted_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    return extracted_path

class TestIMDbMovieRanking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder \
            .appName("IMDb Analysis Test") \
            .master("local[*]") \
            .config("spark.sql.shuffle.partitions", 4) \
            .getOrCreate()

        # IMDb dataset URLs
        cls.output_directory = "test_imdb_datasets"
        cls.ratings_url = "https://datasets.imdbws.com/title.ratings.tsv.gz"
        cls.movies_url = "https://datasets.imdbws.com/title.basics.tsv.gz"
        cls.principals_url = "https://datasets.imdbws.com/title.principals.tsv.gz"

        # Download and extract datasets
        cls.ratings_file = download_and_extract_imdb_data(cls.ratings_url, cls.output_directory, "title.ratings.tsv.gz")
        cls.movies_file = download_and_extract_imdb_data(cls.movies_url, cls.output_directory, "title.basics.tsv.gz")
        cls.principals_file = download_and_extract_imdb_data(cls.principals_url, cls.output_directory, "title.principals.tsv.gz")

        # Define schemas
        cls.ratings_schema = StructType([
            StructField("tconst", StringType(), True),
            StructField("averageRating", FloatType(), True),
            StructField("numVotes", IntegerType(), True)
        ])

        cls.movies_schema = StructType([
            StructField("tconst", StringType(), True),
            StructField("titleType", StringType(), True),
            StructField("primaryTitle", StringType(), True),
            StructField("isAdult", IntegerType(), True),
            StructField("startYear", StringType(), True),
            StructField("endYear", StringType(), True),
            StructField("runtimeMinutes", StringType(), True),
            StructField("genres", StringType(), True)
        ])

        cls.principals_schema = StructType([
            StructField("tconst", StringType(), True),
            StructField("ordering", IntegerType(), True),
            StructField("nconst", StringType(), True),
            StructField("category", StringType(), True),
            StructField("job", StringType(), True),
            StructField("characters", StringType(), True)
        ])

        # Load datasets
        cls.ratings_df = cls.spark.read.csv(cls.ratings_file, schema=cls.ratings_schema, sep="\t", header=True)
        cls.movies_df = cls.spark.read.csv(cls.movies_file, schema=cls.movies_schema, sep="\t", header=True)
        cls.principals_df = cls.spark.read.csv(cls.principals_file, schema=cls.principals_schema, sep="\t", header=True)

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_ranking_logic(self):
        # Filter movies with a minimum of 500 votes
        filtered_ratings = self.ratings_df.filter(col("numVotes") >= 500)

        # Calculate average number of votes
        average_num_votes = filtered_ratings.select(avg("numVotes")).first()[0]

        # Calculate ranking score and get the top 10 movies
        ranked_movies = filtered_ratings.withColumn(
            "rankingScore",
            (col("numVotes") / lit(average_num_votes)) * col("averageRating")
        ).orderBy(desc("rankingScore"))
        print("Top 10 Movies with Ranking Score:")
        ranked_movies.show()

        top_movies = ranked_movies.join(self.movies_df, "tconst") \
            .filter(col("titleType") == "movie") \
            .select("tconst", "primaryTitle", "rankingScore") \
            .limit(10)

        self.assertEqual(top_movies.count(), 10)

        # Collect movie titles for testing
        top_movie_titles = [row["primaryTitle"] for row in top_movies.collect()]

        # Test if top_movie_titles is a non-empty list
        self.assertGreater(len(top_movie_titles), 0)
        print("Top Movie Titles:")  # Displaying in terminal
        top_movies.select("primaryTitle").show()  # Use show() to display the top movies



    def test_credited_persons(self):
        # Get the top 10 movies
        filtered_ratings = self.ratings_df.filter(col("numVotes") >= 500)
        average_num_votes = filtered_ratings.select(avg("numVotes")).first()[0]
        ranked_movies = filtered_ratings.withColumn(
            "rankingScore",
            (col("numVotes") / lit(average_num_votes)) * col("averageRating")
        ).orderBy(desc("rankingScore"))

        top_movies = ranked_movies.join(self.movies_df, "tconst") \
            .filter(col("titleType") == "movie") \
            .select("tconst", "primaryTitle", "rankingScore") \
            .limit(10)

        top_movie_ids = [row["tconst"] for row in top_movies.collect()]

        # Get credited persons for top movies
        top_movies_credits = self.principals_df.filter(col("tconst").isin(top_movie_ids))
        credited_persons = top_movies_credits.groupBy("nconst").agg(count("tconst").alias("count")).orderBy(desc("count"))

        self.assertGreater(credited_persons.count(), 0)
        print("Most Credited Persons for Top Movies:")  # Displaying in terminal
        credited_persons.show()  # Use show() to display credited persons

if __name__ == "__main__":
    unittest.main()

