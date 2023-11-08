# Databricks notebook source
# MAGIC %md
# MAGIC # HuggingFace Hub API
# MAGIC https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.hf_api.ModelInfo

# COMMAND ----------

from huggingface_hub import HfApi, list_models
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pickle


# COMMAND ----------

# Initialize the Hugging Face API
api = HfApi()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exploring Model Info

# COMMAND ----------

model_list=[]

# List all available models
models = api.list_models()

# Print model information
for model in models:
    m = {}
    m["modelId"] = model.modelId
    m["sha"] = model.sha
    m["tags"] = model.tags
    m["pipeline_tag"] = model.pipeline_tag
    m["siblings"] = model.siblings
    m["private"] = model.private
    m["author"] = model.author
    m["config"] = model.config
    m["securityStatus"] = model.securityStatus
    m["downloads"] = model.downloads
    m["language"] = model.language
    m["license"] = model.license
    model_list.append(m)

# COMMAND ----------

df = pd.DataFrame(model_list)

# COMMAND ----------

df.head()

# COMMAND ----------

df['user']= df['modelId'].apply(lambda x: x.split('/')[0] if x.find('/')!=-1 else "None" )


# COMMAND ----------

df['modelName']= df['modelId'].apply(lambda x: x.split('/')[1] if x.find('/')!=-1 else x )

# COMMAND ----------

# MAGIC %md
# MAGIC # Write to pickle

# COMMAND ----------

# write to pickle to save
df_file = open("hf_models", "ab")
pickle.dump(df, df_file)

# COMMAND ----------

# MAGIC %md
# MAGIC # For when to read data back in

# COMMAND ----------

df = pd.read_pickle("./hf_models.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC # EDA

# COMMAND ----------

df.shape

# COMMAND ----------

# Sorting the DataFrame by downloads and selecting the top 20 rows
df_sorted = df.sort_values('downloads', ascending=False).head(20)
df_sorted = df_sorted.sort_values('downloads', ascending=True).head(20)

# Creating a bar chart for the top 20 downloads
plt.figure(figsize=(10, 6))
plt.barh(df_sorted['modelName'], df_sorted['downloads'], color='skyblue', edgecolor='black')
plt.title('Top 20 Models')
plt.xlabel('Total Downloads')
plt.ylabel('Model Name')

# Customizing the x-axis labels to display values in millions
def format_func(value, tick_number):
    if value >= 1e6:
        value = value / 1e6
        return f'{value:.1f}M'
    else:
        return value

plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))
plt.show()

# COMMAND ----------

df.head()

# COMMAND ----------

users = df.groupby("user").sum("downloads").sort_values("downloads", ascending = False).head(20).reset_index()
users = users.sort_values("downloads", ascending = True)

# Creating a bar chart for the top 20 downloads users
plt.figure(figsize=(10, 6))
plt.barh(users['user'], users['downloads'], color='skyblue', edgecolor='black')
plt.title('Top 20 Users')
plt.xlabel('Total Downloads')
plt.ylabel('User Name')

# Customizing the x-axis labels to display values in millions
def format_func(value, tick_number):
    if value >= 1e6:
        value = value / 1e6
        return f'{value:.1f}M'
    else:
        return value

plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))
plt.show()

# COMMAND ----------

df["modelName"].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC # Review popular tags

# COMMAND ----------

df.head()

# COMMAND ----------

df_counts = df["tags"].explode().value_counts()

# COMMAND ----------

df_counts.to_frame().head(10).T

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exploring dataset info

# COMMAND ----------

dataset_list=[]

# List all available dodels
datasets = api.list_datasets()

# Print dodel infordation
for dataset in datasets:
    d = {}
    d["id"] = dataset.id
    d["sha"] = dataset.sha
    d["lastModified"] = dataset.lastModified
    d["tags"] = dataset.tags
    d["siblings"] = dataset.siblings
    d["private"] = dataset.private
    d["author"] = dataset.author
    d["citation"] = dataset.citation
    d["cardData"] = dataset.cardData
    d["downloads"] = dataset.downloads
    dataset_list.append(d)

# COMMAND ----------

data_df = pd.DataFrame(dataset_list)

# COMMAND ----------

data_df.head()

# COMMAND ----------

data_df["author"].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC # Top authors for data

# COMMAND ----------

authors = data_df.groupby("author").sum("downloads").sort_values("downloads", ascending = False).head(20).reset_index()
authors = authors.sort_values("downloads", ascending = True)

# Creating a bar chart for the top 20 downloads authors
plt.figure(figsize=(10, 6))
plt.barh(authors['author'], authors['downloads'], color='skyblue', edgecolor='black')
plt.title('Top 20 authors')
plt.xlabel('Total Downloads')
plt.ylabel('Author Name')

# Customizing the x-axis labels to display values in millions
def format_func(value, tick_number):
    if value >= 1e6:
        value = value / 1e6
        return f'{value:.1f}M'
    else:
        return value

plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))
plt.show()

# COMMAND ----------

data_df["citation"].value_counts()

# COMMAND ----------


