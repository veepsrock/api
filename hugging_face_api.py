# Databricks notebook source
# MAGIC %md
# MAGIC # HuggingFace Hub API
# MAGIC https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.hf_api.ModelInfo

# COMMAND ----------

from huggingface_hub import HfApi, list_models
import pandas as pd

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
    model_list.append(m)

# COMMAND ----------

df = pd.DataFrame(model_list)

# COMMAND ----------

df.shape

# COMMAND ----------

df.head()

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


