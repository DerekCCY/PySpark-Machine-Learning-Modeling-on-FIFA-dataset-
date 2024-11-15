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


# ## Pytorch

# In[32]:

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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[33]:


train_pd = train_processed.toPandas()
test_pd = test_processed.toPandas()
X_train = torch.from_numpy(np.array(train_pd['features'].values.tolist(), np.float32))
y_train = torch.from_numpy(np.array(train_pd['target'].values, np.float32))

X_test = torch.from_numpy(np.array(test_pd['features'].values.tolist(), np.float32))
y_test = torch.from_numpy(np.array(test_pd['target'].values, np.float32))


# In[34]:


import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 
    
set_seed(42)


# In[35]:


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


# In[36]:


train_dataset = MyDataset(X_train, y_train)
test_dataset = MyDataset(X_test, y_test)


# In[37]:


input_sizes = X_train.shape[1]
print(input_sizes)


# ### Model 1: Pytorch Neural Network 1

# In[38]:


class FIFAmodel(nn.Module):
    def __init__(self, ):
        super().__init__()
        input_size = input_sizes
        output_size = 1
        self.lin1 = nn.Linear(input_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, 128)
        self.lin4 = nn.Linear(128, output_size)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.lin1(x))
        x = self.act(self.lin2(x))
        x = self.act(self.lin3(x))
        x = self.lin4(x)
        return x


# In[39]:


model1 = FIFAmodel().to(device)

lr = 1e-3
batch_size = 256
epochs = 100

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model1.parameters(), lr=lr)
scheduler = ExponentialLR(optimizer, gamma=0.9)

training_loss = [] # training losses of each epoch

current_best_loss = float('inf')


# In[40]:


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# In[41]:


training_loss = [] # training losses of each epoch

for epoch in range(epochs):
    training_batch_loss = []
    training_batch_accuracy = []
    for x_batch, y_batch in train_dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        y_pred = model1(x_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_batch_loss.append(loss.detach().cpu().numpy()) # batch loss of training
   
    scheduler.step()
    # training and validation loss of each epoch
    training_loss.append(np.mean(np.array(training_batch_loss)))
    # printing
    if epoch%10 == 0:
        print(f"Epoch = {epoch}, training_loss = {training_loss[-1]}")

    # save the best model
    # if training_loss[-1] < current_best_loss:
    #     torch.save(model1.state_dict(), 'gs://jtang3/fifa_best_model1.pth')
    #     current_best_loss = training_loss[-1]


# In[42]:


# load the best model back
# model1.load_state_dict(torch.load('gs://jtang3/fifa_best_model1.pth'))

# Testing
test_loss = []
model1.eval()
with torch.no_grad():
    for x_batch, y_batch in test_dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        test_pred = model1(x_batch)
        test_loss.append(loss_fn(test_pred, y_batch).detach().cpu().numpy())
    
    print(f"Test loss = {np.round(np.mean(test_loss))}")


# ### Model 2: Pytorch Neural Network 2

# In[43]:


class FIFAmodel2(nn.Module):
    def __init__(self, ):
        super().__init__()
        input_size = input_sizes
        output_size = 1
        self.lin1 = nn.Linear(input_size, 512)
        self.lin2 = nn.Linear(512, 512)
        self.lin3 = nn.Linear(512, 512)
        self.lin4 = nn.Linear(512, 512)
        self.lin5 = nn.Linear(512, 512)
        self.lin6 = nn.Linear(512, 512)
        self.lin7 = nn.Linear(512, 512)
        self.lin8 = nn.Linear(512, 512)
        self.lin9 = nn.Linear(512, output_size)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.lin1(x))
        x = self.act(self.lin2(x))
        x = self.act(self.lin3(x))
        x = self.act(self.lin4(x))
        x = self.act(self.lin5(x))
        x = self.act(self.lin6(x))
        x = self.act(self.lin7(x))
        x = self.act(self.lin8(x))
        x = self.lin9(x)
        return x


# In[44]:


model2 = FIFAmodel2().to(device)

lr = 1e-3
batch_size = 256
epochs = 100

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model2.parameters(), lr=lr)
scheduler = ExponentialLR(optimizer, gamma=0.9)

training_loss = [] # training losses of each epoch

current_best_loss = float('inf')


# In[45]:


training_loss = [] # training losses of each epoch

for epoch in range(epochs):
    training_batch_loss = []
    for x_batch, y_batch in train_dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        y_pred = model2(x_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_batch_loss.append(loss.detach().cpu().numpy()) # batch loss of training
   
    scheduler.step()
    # training and validation loss of each epoch
    training_loss.append(np.mean(np.array(training_batch_loss)))
    # printing
    if epoch%10 == 0:
        print(f"Epoch = {epoch}, training_loss = {training_loss[-1]}")

    # save the best model2
    # if training_loss[-1] < current_best_loss:
    #     torch.save(model2.state_dict(), 'gs://jtang3/fifa_best_model2.pth')
    #     current_best_loss = training_loss[-1]


# In[46]:


# load the best model2 back
# model2.load_state_dict(torch.load('gs://jtang3/fifa_best_model2.pth'))

# Testing
test_loss = []
model2.eval()
with torch.no_grad():
    for x_batch, y_batch in test_dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        test_pred = model2(x_batch)
        test_loss.append(loss_fn(test_pred, y_batch).detach().cpu().numpy())
    
    print(f"Test loss = {np.round(np.mean(test_loss))}")


# In[ ]:




