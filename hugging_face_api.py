# Databricks notebook source
# MAGIC %md
# MAGIC # HuggingFace Hub API
# MAGIC https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.hf_api.ModelInfo

# COMMAND ----------

from huggingface_hub import HfApi, list_models
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


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
    model_list.append(m)

# COMMAND ----------

df = pd.DataFrame(model_list)

# COMMAND ----------

df["author"].unique()

# COMMAND ----------

df.head()

# COMMAND ----------

df["author"].unique()

# COMMAND ----------

df.head()

# COMMAND ----------

df['user']= df['modelId'].apply(lambda x: x.split('/')[0] if x.find('/')!=-1 else "None" )


# COMMAND ----------

df['modelName']= df['modelId'].apply(lambda x: x.split('/')[1] if x.find('/')!=-1 else x )

# COMMAND ----------

# Sorting the DataFrame by downloads and selecting the top 20 rows
df_sorted = df.sort_values('downloads', ascending=False).head(20)

# Creating a bar chart for the top 20 downloads
plt.figure(figsize=(10, 6))
plt.barh(df_sorted['modelName'], df_sorted['downloads'], color='skyblue', edgecolor='black')
plt.title('Top 20 Downloads by Model Name')
plt.xlabel('Model Name')
plt.ylabel('Downloads')

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

df_sorted

# COMMAND ----------

df["modelId"].str.split("/", expand = True)

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
    dataset_list.append(d)

# COMMAND ----------

data_df = pd.DataFrame(dataset_list)

# COMMAND ----------

data_df.head()

# COMMAND ----------

data_df.shape

# COMMAND ----------


