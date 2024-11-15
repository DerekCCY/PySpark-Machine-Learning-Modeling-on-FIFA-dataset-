#!/usr/bin/env python
# coding: utf-8



# In[2]:


import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkContext, SQLContext
from pyspark.sql import functions as F
# Ensure the correct import
from pyspark.sql.functions import col, when, count
from pyspark.sql.types import DoubleType

from pyspark.ml import Pipeline,Transformer
from pyspark.ml.feature import StandardScaler,StringIndexer,OneHotEncoder, VectorAssembler
from pyspark.sql.functions import *
from pyspark.sql.types import *

import os
import sys  
import subprocess

from functools import reduce

try:
    import torch
    print("successfully import torch")
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
    import torch
    print("Successfully download torch and import it")
# In[3]:

appName = 'fifa'
spark = SparkSession.builder \
    .appName(appName) \
    .master("yarn") \
    .getOrCreate()


# ## Read data from PostgreSQL database

# In[4]:

# players_df_csv = players_df.toPandas().to_csv('players_df.csv', index=False, header=True)
players_df = spark.read.csv('gs://jtang3/players_df.csv', header=True, inferSchema=True)


# ## Data Cleaning

# ### Dealing with missing values

# In[7]:


na_counts = players_df.select([count(when(col(c).isNull() | (col(c) == "NA"), c)).alias(c) for c in players_df.columns])
# na_counts.show()


# ### Thresholding: drop the columns that contains over 50% missing value

# In[8]:


from pyspark.sql.functions import col, when, count

total_rows = players_df.count()
na_counts_df = players_df.select([count(when(col(c).isNull() | (col(c) == "NA"), c)).alias(c) for c in players_df.columns])
na_ratio = {c: na_counts_df.collect()[0][c] / total_rows for c in players_df.columns}
columns_to_drop = [c for c, ratio in na_ratio.items() if ratio > 0.5]
df_cleaned = players_df.drop(*columns_to_drop)


# ## Feature Selection and Engineering

# ### Select the column that are meaningful for the overall prediction

# In[9]:


selected_columns = [
    "overall", "potential", "age", "height_cm", "weight_kg",
    "preferred_foot", "weak_foot", "skill_moves", "international_reputation", "pace", "shooting", "passing",
    "dribbling", "defending", "physic", "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy",
    "attacking_short_passing", "attacking_volleys", "skill_dribbling", "skill_curve", "skill_fk_accuracy",
    "skill_long_passing", "skill_ball_control", "movement_acceleration", "movement_sprint_speed", "movement_agility",
    "movement_reactions", "movement_balance", "power_shot_power", "power_jumping", "power_stamina", "power_strength",
    "power_long_shots", "mentality_aggression", "mentality_interceptions", "mentality_positioning", "mentality_vision",
    "mentality_penalties", "mentality_composure", "defending_marking_awareness", "defending_standing_tackle",
    "defending_sliding_tackle", "goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking",
    "goalkeeping_positioning", "goalkeeping_reflexes", "ls", "st", "rs", "lw", "lf", "cf", "rf", "rw", "lam", "cam",
    "ram", "lm", "lcm", "cm", "rcm", "rm", "lwb", "ldm", "cdm", "rdm", "rwb", "lb", "lcb", "cb", "rcb", "rb", "gk",
    "year", "gender"
]

df_selected = df_cleaned.select(*selected_columns)

# df_selected.show()


# ### Interpolate Null values with average value

# In[10]:


null_list = ["pace", "shooting", "passing", "dribbling", "defending", "physic", "mentality_composure"]

df_interpolated = df_selected

for column in null_list:
    avg_value = df_interpolated.select(mean(col(column))).collect()[0][0]
    
    df_interpolated = df_interpolated.withColumn(
        column,
        when(col(column).isNull(), avg_value).otherwise(col(column)).cast('double')
    )


# In[11]:


na_counts = df_interpolated.select([count(when(col(c).isNull() | (col(c) == "NA"), c)).alias(c) for c in df_interpolated.columns])
# na_counts.show(vertical = True)


# In[12]:


# df_interpolated.show()


# ### Convert the calculation (string dtype) into integer

# In[13]:


columns_to_convert = [
    "ls", "st", "rs", 'lw', 'lf', 'cf', 'rf', 'rw', "lam", "cam", "ram", "lm", "lcm", "cm", "rcm", "rm", 
    "lwb", "ldm", "cdm", "rdm", "rwb", "lb", "lcb", "cb", "rcb", "rb", "gk"
]

# 使用正則表達式來只保留符號前的數字
for col in columns_to_convert:
    df_interpolated = df_interpolated.withColumn(
        col,
        F.when(
            F.col(col).rlike(r"[+-]"),
            F.expr(f"cast(regexp_extract({col}, '^[0-9]+', 0) as int)")
        ).otherwise(F.col(col).cast('int'))
    )

# 查看結果
# df_interpolated.show(5)
df_result = df_interpolated


# In[14]:


null_counts = df_result.select([count(F.when(F.col(c).isNull() | isnan(F.col(c)) | (F.col(c) == "NA"), c)).alias(c) for c in df_result.columns])
# null_counts.show(vertical=True)


# ### Convert gender and preferred_foot into 0 and 1
# ### Convert mentaility_composure from string to integer

# In[15]:


# Convert "M" to 1 and "F" to 0 for the gender column, allowing nulls if value is neither
df_result = df_result.withColumn(
    "gender",
    when(F.col("gender") == "M", 1).when(F.col("gender") == "F", 0).otherwise(F.lit(None)).cast("int")
)

# Convert "Right" to 1 and "Left" to 0 for the preferred_foot column, allowing nulls if value is neither
df_result = df_result.withColumn(
    "preferred_foot",
    when(F.col("preferred_foot") == "Right", 1).when(F.col("preferred_foot") == "Left", 0).otherwise(F.lit(None)).cast("int")
)
df_result = df_result.withColumn(
    "mentality_composure",
    F.col("mentality_composure").cast('int')
)


# In[16]:


non_numeric_counts = df_result.select([
    count(when(~F.col(c).cast("double").isNotNull() & F.col(c).isNotNull(), c)).alias(c)
    for c in df_result.columns
])

# Show the results to identify any columns with non-numeric values
# non_numeric_counts.show()


# In[17]:


df_result.printSchema()


# ### Outliers removal

# In[18]:



def column_add(a,b):
    return  a.__add__(b)
    
def find_outliers(df):
    # Identifying the numerical columns in a spark dataframe
    numeric_columns = [column[0] for column in df.dtypes if column[1]=='int']

    # Using the `for` loop to create new columns by identifying the outliers for each feature
    for column in numeric_columns:

        less_Q1 = 'less_Q1_{}'.format(column)
        more_Q3 = 'more_Q3_{}'.format(column)
        Q1 = 'Q1_{}'.format(column)
        Q3 = 'Q3_{}'.format(column)

        # Q1 : First Quartile ., Q3 : Third Quartile
        Q1 = df.approxQuantile(column,[0.25],relativeError=0)
        Q3 = df.approxQuantile(column,[0.75],relativeError=0)
        
        # IQR : Inter Quantile Range
        # We need to define the index [0], as Q1 & Q3 are a set of lists., to perform a mathematical operation
        # Q1 & Q3 are defined seperately so as to have a clear indication on First Quantile & 3rd Quantile
        IQR = Q3[0] - Q1[0]
        
        #selecting the data, with -1.5*IQR to + 1.5*IQR., where param = 1.5 default value
        less_Q1 =  Q1[0] - 1.5*IQR
        more_Q3 =  Q3[0] + 1.5*IQR
        
        isOutlierCol = 'is_outlier_{}'.format(column)
        
        df = df.withColumn(isOutlierCol,when((df[column] > more_Q3) | (df[column] < less_Q1), 1).otherwise(0))
    

    # Selecting the specific columns which we have added above, to check if there are any outliers
    selected_columns = [column for column in df.columns if column.startswith("is_outlier")]
    # Adding all the outlier columns into a new colum "total_outliers", to see the total number of outliers
    df = df.withColumn('total_outliers',reduce(column_add, ( df[col] for col in  selected_columns)))

    # Dropping the extra columns created above, just to create nice dataframe., without extra columns
    df = df.drop(*[column for column in df.columns if column.startswith("is_outlier")])

    return df


# In[19]:


numeric_features = [feature[0] for feature in df_result.dtypes if feature[1] in ('int','double')]
df_numeric = df_result.select(numeric_features)


# In[20]:


df_with_outlier_handling = find_outliers(df_numeric)
# df_with_outlier_handling.groupby("total_outliers").count().show()


# ### Visualize outliers

# In[21]:


# import matplotlib.pyplot as plt

# df_pandas = df_with_outlier_handling.select("total_outliers").toPandas()

# plt.figure(figsize=(10, 6))
# plt.hist(df_pandas["total_outliers"], bins=20, edgecolor='black', color='skyblue')
# plt.title("Histogram of Total Outliers")
# plt.xlabel("Total Outliers")
# plt.ylabel("Frequency")
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# plt.show()


# ### Remove the rows which have more than ten outliers

# In[22]:


df_with_substituted_outliers = df_with_outlier_handling.\
filter(df_with_outlier_handling['total_outliers']<=10)
        
df_with_substituted_outliers = df_with_substituted_outliers.drop("total_outliers")


# In[23]:


final_df  = df_with_substituted_outliers


# store train_processed df as csv file
final_df_csv = final_df.toPandas().to_csv('gs://jtang3/final_df.csv', index=False, header=True)
# ### Correlation plot

# In[24]:


# import seaborn as sns
# correlation_matrix = final_df.toPandas().corr()
# plt.figure(figsize=(18, 16))
# sns.heatmap(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1, annot=False, fmt=".2f", square=True)
# plt.title("Improved Correlation Matrix Heatmap")
# plt.show()


# ### Train_test split

# In[25]:


train_data, test_data = final_df.randomSplit([0.8, 0.2], seed=66)


# ### Pipeline

# In[26]:


nominal_cols = [
    "preferred_foot", "gender"
]

continuous_cols = [
    "potential", "age", "height_cm", "weight_kg",  
    "weak_foot", "skill_moves", "international_reputation", "pace", "shooting", 
    "passing", "dribbling", "defending", "physic", "attacking_crossing", 
    "attacking_finishing", "attacking_heading_accuracy", "attacking_short_passing", 
    "attacking_volleys", "skill_dribbling", "skill_curve", "skill_fk_accuracy", 
    "skill_long_passing", "skill_ball_control", "movement_acceleration", 
    "movement_sprint_speed", "movement_agility", "movement_reactions", 
    "movement_balance", "power_shot_power", "power_jumping", "power_stamina", 
    "power_strength", "power_long_shots", "mentality_aggression", 
    "mentality_interceptions", "mentality_positioning", "mentality_vision", 
    "mentality_penalties", "mentality_composure", "defending_marking_awareness", "defending_standing_tackle", 
    "defending_sliding_tackle", "goalkeeping_diving", "goalkeeping_handling", 
    "goalkeeping_kicking", "goalkeeping_positioning", "goalkeeping_reflexes",
    "ls", "st", "rs", "lw", "lf", "cf", "rf", "rw", "lam", "cam", "ram", 
    "lm", "lcm", "cm", "rcm", "rm", "lwb", "ldm", "cdm", "rdm", "rwb", 
    "lb", "lcb", "cb", "rcb", "rb", "gk", "year"
]


# In[27]:


class OutcomeCreater(Transformer): # this defines a transformer that creates the outcome column
    
    def __init__(self):
        super().__init__()

    def _transform(self, dataset):
        output_df = dataset.withColumn('target', F.col('overall').cast(DoubleType())).drop("overall")
        return output_df

class FeatureTypeCaster(Transformer): # this transformer will cast the columns as appropriate types  
    def __init__(self):
        super().__init__()

    def _transform(self, dataset):
        output_df = dataset
        for col_name in continuous_cols:
            output_df = output_df.withColumn(col_name,F.col(col_name).cast(DoubleType()))

        return output_df

class ColumnDropper(Transformer): # this transformer drops unnecessary columns
    def __init__(self, columns_to_drop = None):
        super().__init__()
        self.columns_to_drop=columns_to_drop
    def _transform(self, dataset):
        output_df = dataset
        for col_name in self.columns_to_drop:
            output_df = output_df.drop(col_name)
        return output_df    

def get_preprocess_pipeline():
    # Stage where columns are casted as appropriate types
    stage_typecaster = FeatureTypeCaster()

    # Stage where nominal columns are transformed to index columns using StringIndexer
    nominal_id_cols = [x + "_index" for x in nominal_cols]
    nominal_onehot_cols = [x + "_encoded" for x in nominal_cols]
    stage_nominal_indexer = StringIndexer(inputCols=nominal_cols, outputCols=nominal_id_cols)

    # Stage where the index columns are further transformed using OneHotEncoder
    stage_nominal_onehot_encoder = OneHotEncoder(inputCols=nominal_id_cols, outputCols=nominal_onehot_cols)

    # Stage where all relevant features are assembled into a vector (and dropping a few)
    feature_cols = continuous_cols  + nominal_onehot_cols

    stage_vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="vectorized_features")

    # Stage where we scale the columns
    stage_scaler = StandardScaler(inputCol='vectorized_features', outputCol='features')

    # Stage for creating the outcome column representing the 5-class label
    stage_outcome = OutcomeCreater()

    # Removing all unnecessary columns, only keeping the 'features' and 'target' columns
    stage_column_dropper = ColumnDropper(columns_to_drop=nominal_cols + nominal_id_cols +
                                         nominal_onehot_cols  + continuous_cols + 
                                         ['vectorized_features'])

    # Connect the stages into a pipeline
    pipeline = Pipeline(stages=[stage_typecaster, stage_nominal_indexer, stage_nominal_onehot_encoder,
                                stage_vector_assembler, stage_scaler, stage_outcome,
                                stage_column_dropper])
    
    return pipeline


# In[28]:


# establish training pipeline
pipeline = get_preprocess_pipeline()

# fit the model with pipeline
preprocessing = pipeline.fit(train_data)

# make predictions on testing dataset
train_processed = preprocessing.transform(train_data)
test_processed = preprocessing.transform(test_data)

# print testing dataset pipeline
train_processed.orderBy(F.col('target').desc()).show()


# In[29]:


# # Convert Spark DataFrame to Pandas DataFrame
# pandas_df = train_processed.toPandas()

# # Display the full feature list of the first row
# print(len(list(pandas_df['features'][0])))
# print(list(pandas_df['features'][0]))

