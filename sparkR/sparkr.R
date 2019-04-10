
# data from https://github.com/apache/spark/tree/master/data/mllib


# https://spark.apache.org/docs/latest/sparkr.html#overview

# Overview
# 
# SparkR is an R package that provides a light-weight frontend to use Apache Spark from R. In Spark 2.4.1, SparkR provides a distributed data frame implementation that supports operations like selection, filtering, aggregation etc. (similar to R data frames, dplyr) but on large datasets. SparkR also supports distributed machine learning using MLlib.
# SparkDataFrame
# 
# A SparkDataFrame is a distributed collection of data organized into named columns. It is conceptually equivalent to a table in a relational database or a data frame in R, but with richer optimizations under the hood. SparkDataFrames can be constructed from a wide array of sources such as: structured data files, tables in Hive, external databases, or existing local R data frames.
# 
# All of the examples on this page use sample data included in R or the Spark distribution and can be run using the ./bin/sparkR shell.

install.packages("SparkR")
library(SparkR)
sparkR.session()

if (nchar(Sys.getenv("SPARK_HOME")) < 1) {
  Sys.setenv(SPARK_HOME = "/home/spark")
}
library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))
sparkR.session(master = "local[*]", sparkConfig = list(spark.driver.memory = "2g"))


# Creating SparkDataFrames  ----
# From local data frames
df <- as.DataFrame(faithful)
# Displays the first part of the SparkDataFrame
head(df)

# From Data Sources
sparkR.session(sparkPackages = "com.databricks:spark-avro_2.11:3.0.0")

people <- read.df("people.json", "json")
head(people)
# SparkR automatically infers the schema from the JSON file
printSchema(people)
# Similarly, multiple files can be read with read.json
people <- read.json(c("./examples/src/main/resources/people.json", "./examples/src/main/resources/people2.json"))

# The data sources API natively supports CSV formatted input files. For more information please refer to SparkR read.df API documentation.

df <- read.df(csvPath, "csv", header = "true", inferSchema = "true", na.strings = "NA")

# The data sources API can also be used to save out SparkDataFrames into multiple file formats. For example, we can save the SparkDataFrame from the previous example to a Parquet file using write.df.

write.df(people, path = "people.parquet", source = "parquet", mode = "overwrite")

# From Hive tables
sparkR.session()

sql("CREATE TABLE IF NOT EXISTS src (key INT, value STRING)")
sql("LOAD DATA LOCAL INPATH 'examples/src/main/resources/kv1.txt' INTO TABLE src")

# Queries can be expressed in HiveQL.
results <- sql("FROM src SELECT key, value")

# results is now a SparkDataFrame
head(results)


# SparkDataFrame Operations  ----

# SparkDataFrames support a number of functions to do structured data processing. Here we include some basic examples and a complete list can be found in the API docs:

# Selecting rows, columns

# Create the SparkDataFrame
df <- as.DataFrame(faithful)

# Get basic information about the SparkDataFrame
df
## SparkDataFrame[eruptions:double, waiting:double]

# Select only the "eruptions" column
head(select(df, df$eruptions))
##  eruptions
##1     3.600
##2     1.800
##3     3.333

# You can also pass in column name as strings
head(select(df, "eruptions"))

# Filter the SparkDataFrame to only retain rows with wait times shorter than 50 mins
head(filter(df, df$waiting < 50))
##  eruptions waiting

# Grouping, Aggregation

# SparkR data frames support a number of commonly used functions to aggregate data after grouping. For example, we can compute a histogram of the waiting time in the faithful dataset as shown below

# We use the `n` operator to count the number of times each waiting time appears
head(summarize(groupBy(df, df$waiting), count = n(df$waiting)))

# We can also sort the output from the aggregation to get the most common waiting times
waiting_counts <- summarize(groupBy(df, df$waiting), count = n(df$waiting))
head(arrange(waiting_counts, desc(waiting_counts$count)))

# In addition to standard aggregations, SparkR supports OLAP cube operators cube:

head(agg(cube(df, "cyl", "disp", "gear"), avg(df$mpg)))

# and rollup:

head(agg(rollup(df, "cyl", "disp", "gear"), avg(df$mpg)))


# Operating on Columns ----

# SparkR also provides a number of functions that can directly applied to columns for data processing and during aggregation. The example below shows the use of basic arithmetic functions.

# Convert waiting time from hours to seconds.
# Note that we can assign this to a new column in the same SparkDataFrame
df$waiting_secs <- df$waiting * 60
head(df)


# Applying User-Defined Function -----

# Run a given function on a large dataset using dapply or dapplyCollect

# Convert waiting time from hours to seconds.
# Note that we can apply UDF to DataFrame.
schema <- structType(structField("eruptions", "double"), structField("waiting", "double"),
                     structField("waiting_secs", "double"))
df1 <- dapply(df, function(x) { x <- cbind(x, x$waiting * 60) }, schema)
head(collect(df1))


# Convert waiting time from hours to seconds.
# Note that we can apply UDF to DataFrame and return a R's data.frame
ldf <- dapplyCollect(
         df,
         function(x) {
           x <- cbind(x, "waiting_secs" = x$waiting * 60)
         })
head(ldf, 3)


# Run a given function on a large dataset grouping by input column(s) and using gapply or gapplyCollect

# Determine six waiting times with the largest eruption time in minutes.
schema <- structType(structField("waiting", "double"), structField("max_eruption", "double"))
result <- gapply(
    df,
    "waiting",
    function(key, x) {
        y <- data.frame(key, max(x$eruptions))
    },
    schema)
head(collect(arrange(result, "max_eruption", decreasing = TRUE)))

# Determine six waiting times with the largest eruption time in minutes.
result <- gapplyCollect(
    df,
    "waiting",
    function(key, x) {
        y <- data.frame(key, max(x$eruptions))
        colnames(y) <- c("waiting", "max_eruption")
        y
    })
head(result[order(result$max_eruption, decreasing = TRUE), ])

# Run local R functions distributed using spark.lapply 
# Perform distributed training of multiple models with spark.lapply. Here, we pass
# a read-only list of arguments which specifies family the generalized linear model should be.
families <- c("gaussian", "poisson")
train <- function(family) {
  model <- glm(Sepal.Length ~ Sepal.Width + Species, iris, family = family)
  summary(model)
}
# Return a list of model's summaries
model.summaries <- spark.lapply(families, train)

# Print the summary of each model
print(model.summaries)


# Running SQL Queries from SparkR  ----

# A SparkDataFrame can also be registered as a temporary view in Spark SQL and that allows you to run SQL queries over its data. The sql function enables applications to run SQL queries programmatically and returns the result as a SparkDataFrame.

# Load a JSON file
people <- read.df("people.json", "json")

# Register this SparkDataFrame as a temporary view.
createOrReplaceTempView(people, "people")

# SQL statements can be run by using the sql method
teenagers <- sql("SELECT name FROM people WHERE age >= 13 AND age <= 19")
head(teenagers)


# Machine Learning  ------
Algorithms

SparkR supports the following machine learning algorithms currently:
Classification

    spark.logit: Logistic Regression
    spark.mlp: Multilayer Perceptron (MLP)
    spark.naiveBayes: Naive Bayes
    spark.svmLinear: Linear Support Vector Machine

Regression

    spark.survreg: Accelerated Failure Time (AFT) Survival Model
    spark.glm or glm: Generalized Linear Model (GLM)
    spark.isoreg: Isotonic Regression

Tree

    spark.decisionTree: Decision Tree for Regression and Classification
    spark.gbt: Gradient Boosted Trees for Regression and Classification
    spark.randomForest: Random Forest for Regression and Classification

Clustering

    spark.bisectingKmeans: Bisecting k-means
    spark.gaussianMixture: Gaussian Mixture Model (GMM)
    spark.kmeans: K-Means
    spark.lda: Latent Dirichlet Allocation (LDA)

Collaborative Filtering

    spark.als: Alternating Least Squares (ALS)

Frequent Pattern Mining

    spark.fpGrowth : FP-growth

Statistics

    spark.kstest: Kolmogorov-Smirnov Test

Under the hood, SparkR uses MLlib to train the model. Please refer to the corresponding section of MLlib user guide for example code. Users can call summary to print a summary of the fitted model, predict to make predictions on new data, and write.ml/read.ml to save/load fitted models. SparkR supports a subset of the available R formula operators for model fitting, including ‘~’, ‘.’, ‘:’, ‘+’, and ‘-‘.
Model persistence

# The following example shows how to save/load a MLlib model by SparkR.

training <- read.df("sample_multiclass_classification_data.txt", source = "libsvm")
# Fit a generalized linear model of family "gaussian" with spark.glm
df_list <- randomSplit(training, c(7,3), 2)
gaussianDF <- df_list[[1]]
gaussianTestDF <- df_list[[2]]
gaussianGLM <- spark.glm(gaussianDF, label ~ features, family = "gaussian")

# Save and then load a fitted MLlib model
modelPath <- tempfile(pattern = "ml", fileext = ".tmp")
write.ml(gaussianGLM, modelPath)
gaussianGLM2 <- read.ml(modelPath)

# Check model summary
summary(gaussianGLM2)

# Check model prediction
gaussianPredictions <- predict(gaussianGLM2, gaussianTestDF)
head(gaussianPredictions)

unlink(modelPath)

















