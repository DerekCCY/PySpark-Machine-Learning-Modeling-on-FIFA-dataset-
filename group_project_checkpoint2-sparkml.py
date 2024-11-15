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


# In[3]:

appName = 'fifa'
spark = SparkSession.builder \
    .appName(appName) \
    .master("yarn") \
    .getOrCreate()

# ## Training

# ### Model1: Spark Linear Regression

# In[30]:


final_df = spark.read.csv('gs://jtang3/final_df.csv', header=True, inferSchema=True)
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

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
model_lr = LinearRegression(featuresCol='features', labelCol='target', maxIter=100)

# Fit the model

model_lr = model_lr.fit(train_processed)
train_predictions = model_lr.transform(train_processed)
# train_predictions.show()

test_predictions = model_lr.transform(test_processed)
# test_predictions.show()
evaluator = RegressionEvaluator(labelCol="target", predictionCol="prediction", metricName="mse")
train_mse = evaluator.evaluate(train_predictions)
test_mse = evaluator.evaluate(test_predictions)
print(f"Train RMSE: {train_mse:.2f}")
print(f"Test RMSE: {test_mse:.2f}")


# ### Model 2: Spark Decision Tree

# In[31]:


from pyspark.ml.regression import DecisionTreeRegressor,RandomForestRegressor
model_dt = DecisionTreeRegressor(featuresCol='features', labelCol='target', maxDepth=12)

# Fit the model

model_dt = model_dt.fit(train_processed)
train_predictions = model_dt.transform(train_processed)
train_predictions.show()

test_predictions = model_dt.transform(test_processed)
test_predictions.show()
evaluator = RegressionEvaluator(labelCol="target", predictionCol="prediction", metricName="mse")
train_mse = evaluator.evaluate(train_predictions)
test_mse = evaluator.evaluate(test_predictions)
print(f"Train MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")