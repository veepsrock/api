# Databricks notebook source
import os
import requests
import pandas as pd

# COMMAND ----------

token = os.environ['github_token']

# COMMAND ----------

# Define the base URL for the GitHub API
base_url = 'https://api.github.com'

# Define the endpoints for repositories and organizations
repositories_endpoint = '/repositories'
organizations_endpoint = '/organizations'

# COMMAND ----------

# Set up headers with your authentication token
headers = {
    'Authorization': f'token {token}'
}

# COMMAND ----------

# Make a GET request to list all public repositories
repositories_url = f'{base_url}{repositories_endpoint}'
repositories_response = requests.get(repositories_url, headers=headers)

if repositories_response.status_code == 200:
    # Parse the JSON response
    repositories_data = repositories_response.json()

    # Iterate through the repositories and print basic information
    for repo in repositories_data:
        print(f"Repository Name: {repo['name']}")
        print(f"Owner: {repo['owner']['login']}")
        print(f"Description: {repo['description']}\n")

# COMMAND ----------

df = pd.DataFrame(repositories_data)

# COMMAND ----------

df.shape

# COMMAND ----------

df.head()

# COMMAND ----------

df.columns

# COMMAND ----------

df["description"]

# COMMAND ----------

# Make a GET request to list all public organizations
organizations_url = f'{base_url}{organizations_endpoint}'
organizations_response = requests.get(organizations_url, headers=headers)

if organizations_response.status_code == 200:
    # Parse the JSON response
    organizations_data = organizations_response.json()

    # Iterate through the organizations and print basic information
    for org in organizations_data:
        print(f"Organization Name: {org['login']}")
        print(f"Description: {org['description']}\n")
else:
    print(f"Failed to fetch organizations. Status code: {organizations_response.status_code}")

# COMMAND ----------

orgs = pd.DataFrame(organizations_data)

# COMMAND ----------

orgs.shape

# COMMAND ----------

orgs

# COMMAND ----------


