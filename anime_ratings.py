from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col

# Iniciar SparkSession
spark = SparkSession.builder.appName("AnimeRatings").getOrCreate()

# Archivos
ratings_file = "../resources/data/als/ratings.csv"
anime_file = "../resources/data/als/anime.csv"
test_file = "../resources/data/als/test.data"

# Leer los archivos CSV
ratings_df = spark.read.option("header", "true").csv(ratings_file)
anime_df = spark.read.option("header", "true").csv(anime_file)
test_df = spark.read.option("header", "true").csv(test_file)

# Convertir columnas a tipos correctos
ratings_df = ratings_df.withColumn("userId", col("user_id").cast("int")) \
                       .withColumn("anime_id", col("anime_id").cast("int")) \
                       .withColumn("rating", col("rating").cast("float"))

anime_df = anime_df.withColumn("anime_id", col("anime_id").cast("int"))

test_df = test_df.withColumn("userId", col("user_id").cast("int")) \
                 .withColumn("anime_id", col("anime_id").cast("int")) \
                 .withColumn("rating", col("rating").cast("float"))

# Verificar los tipos de columnas
ratings_df.printSchema()
anime_df.printSchema()
test_df.printSchema()

# Número de ratings, usuarios y animes
num_ratings = ratings_df.count()
num_users = ratings_df.select("userId").distinct().count()
num_animes = ratings_df.select("anime_id").distinct().count()

print(f"Got {num_ratings} ratings from {num_users} users on {num_animes} animes.")

# Definir y entrenar el modelo ALS
als = ALS(userCol="userId", itemCol="anime_id", ratingCol="rating", rank=10, maxIter=10, coldStartStrategy="drop")

# Entrenar el modelo
model = als.fit(ratings_df)

# Hacer predicciones en el conjunto de prueba
predictions_df = model.transform(test_df)
predictions_df.show(20)

# Filtrar false positives
false_positives = predictions_df.alias("predictions") \
    .join(anime_df.alias("anime"), col("predictions.anime_id") == col("anime.anime_id")) \
    .filter((col("predictions.rating") <= 1) & (col("predictions.prediction") >= 4)) \
    .select(
        col("predictions.userId"),
        col("predictions.anime_id"),
        col("anime.name"),
        col("predictions.rating"),
        col("predictions.prediction")
    )
false_positives.show()

# Guardar el modelo
model.save("./save/anime_ratings_model")
