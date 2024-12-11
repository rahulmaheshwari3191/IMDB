import unittest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit

class TestStreamingRanking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder \
            .appName("IMDb Analysis Test - Streaming") \
            .master("local[*]") \
            .getOrCreate()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_streaming_ranking(self):
        # Prepare streaming source
        movies_stream_data = [
            {"tconst": "tt001", "numVotes": 1000, "averageRating": 8.5},
            {"tconst": "tt002", "numVotes": 2000, "averageRating": 9.0},
            {"tconst": "tt003", "numVotes": 500, "averageRating": 7.0}
        ]

        # Create a DataFrame from the sample data
        static_df = self.spark.createDataFrame(movies_stream_data)

        # Register the static DataFrame as a temporary streaming input
        static_df.createOrReplaceTempView("movies_temp")
        streaming_df = self.spark.sql("SELECT * FROM movies_temp")

        # Simulate ranking logic
        average_num_votes = 1500  # Predefined average for testing
        ranked_movies_stream = streaming_df.withColumn(
            "rankingScore",
            (col("numVotes") / lit(average_num_votes)) * col("averageRating")
        ).orderBy("rankingScore", ascending=False)

        # Memory sink for testing
        query = ranked_movies_stream.writeStream \
            .outputMode("complete") \
            .format("memory") \
            .queryName("test_ranking") \
            .start()

        query.processAllAvailable()  # Trigger the computation

        # Collect results from memory sink
        result_df = self.spark.sql("SELECT * FROM test_ranking")
        results = result_df.collect()

        # Assertions
        self.assertEqual(results[0]["tconst"], "tt002")  # Highest ranked movie
        self.assertEqual(results[1]["tconst"], "tt001")
        self.assertEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()
