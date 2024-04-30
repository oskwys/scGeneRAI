# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:12:11 2024

@author: owysocky
"""
import pandas as pd
import requests
import json
import os

def fetch_data_from_api(url):
    """
    Fetch data from the provided API URL and return it as a Python dictionary.
    If the request fails, return an error message and status code.

    :param url: URL of the API endpoint
    :return: Python dictionary containing the JSON data or error message
    """
    try:
        # Making a GET request to the API
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the response to JSON and return as a dictionary
            return response.json()
        else:
            return {"error": f"Failed to retrieve data. Status code: {response.status_code}"}

    except requests.RequestException as e:
        return {"error": f"An error occurred: {e}"}

# Example usage
url = 'https://pantherdb.org/services/oai/pantherdb/supportedannotdatasets'
print(fetch_data_from_api(url))


def genes_to_string(genes):
    """
    Convert a set of gene names into a string, separated by '%'.

    :param genes: A set of gene names
    :return: A string with each gene name separated by '%'
    """
    return ','.join(genes)

def process_enrichment_results(data_from_api):
    """
    Process the enrichment results obtained from an API.

    :param data_from_api: Data downloaded from the API
    :return: Processed DataFrame with term_id and term_label columns
    """
    # Extracting the 'result' part from the API data
    enrichment_results = data_from_api.get('results', {}).get('result', [])
    
    # Converting the data to a DataFrame
    enrichment_results_df = pd.DataFrame(enrichment_results)

    # Extracting 'id' and 'label' from the 'term' column and adding them as new columns
    if 'term' in enrichment_results_df.columns:
        enrichment_results_df[['term_id', 'term_label']] = enrichment_results_df['term'].apply(pd.Series)[['id', 'label']]

        # Dropping the original 'term' column
        enrichment_results_df.drop(columns=['term'], inplace=True)

    return enrichment_results_df

def create_pantherdb_url(genes, organismLabel, annDatasetLabel, enrichmentTestType, correction, supportedgenomes, annDataSets):
    """
    Create a URL for the PantherDB enrichment analysis, using labels for organism and annotation dataset.

    :param genes: A list of genes
    :param organismLabel: Label of the organism (e.g., 'human')
    :param annDatasetLabel: Label of the annotation dataset (e.g., 'molecular_function')
    :param enrichmentTestType: Type of enrichment test
    :param correction: Correction method
    :param supportedgenomes: DataFrame containing supported genomes and their taxon IDs
    :param annDataSets: DataFrame containing annotation datasets and their IDs
    :return: Formatted URL as a string
    """
    geneInputList = ','.join(genes)
    organism_id = str(supportedgenomes.loc[supportedgenomes['name'] == organismLabel, 'taxon_id'].iloc[0])
    annDatasetId = str(annDataSets.loc[annDataSets['label'] == annDatasetLabel, 'id'].iloc[0]).replace(':', '%3A')

    url = f'https://pantherdb.org/services/oai/pantherdb/enrich/overrep?geneInputList={geneInputList}&organism={organism_id}&annotDataSet={annDatasetId}&enrichmentTestType={enrichmentTestType}&correction={correction}'
    return url


def filter_enrichment_results(data_from_api, pvalue_threshold=0.05):
    """
    Process and filter enrichment results based on a p-value threshold.

    :param data_from_api: Data downloaded from the API
    :param pvalue_threshold: Threshold for filtering based on p-value
    :return: Filtered DataFrame
    """
    # Process the enrichment results
    enrichment_results = process_enrichment_results(data_from_api)

    # Filter the results based on the p-value threshold
    enrichment_results_filtered = enrichment_results[enrichment_results['pValue'] < pvalue_threshold]

    return enrichment_results_filtered


# %% Annotation Data Sets

url = 'https://pantherdb.org/services/oai/pantherdb/supportedannotdatasets'
data_from_api = fetch_data_from_api(url)

annDataSets = data_from_api['search']['annotation_data_sets']['annotation_data_type']
annDataSets = pd.DataFrame(annDataSets)

# %% supportedgenomes (organism )

url = 'https://pantherdb.org/services/oai/pantherdb/supportedgenomes'
data_from_api = fetch_data_from_api(url)

supportedgenomes = data_from_api['search']['output']['genomes']['genome']
supportedgenomes = pd.DataFrame(supportedgenomes)

# %% PANTHER Tools - Enrichment (Overrepresentation)
genes = ['ACHE' ,'AMOT', 'CDK5R1' ,'CDK6', 'CELSR1']
geneInputList = genes_to_string(genes)
organism = '9606'
annotDataSet = 'GO%3A0008150'

enrichmentTestType = 'FISHER'
correction = 'FDR'


url = f'https://pantherdb.org/services/oai/pantherdb/enrich/overrep?geneInputList={geneInputList}&organism={organism}&annotDataSet={annotDataSet}&enrichmentTestType={enrichmentTestType}&correction={correction}'
data_from_api = fetch_data_from_api(url)
enrichment_results = process_enrichment_results(data_from_api)



pvalue_threshold = 0.05
enrichment_results_filtered = enrichment_results[enrichment_results['pValue']< pvalue_threshold]







# %%
path_to_save = r'G:\My Drive\SAFE_AI\CCE_DART\Lunarverse_results'
file = r'C:\Users\owysocky\Desktop\gene_set.csv'
df = pd.read_csv(file)
genes =  df['gene_name'].to_list()#['ACHE' ,'AMOT', 'CDK5R1' ,'CDK6', 'CELSR1']



organismLabel = 'human'
annDatasetLabel = 'molecular_function'
url = create_pantherdb_url(genes, organismLabel, annDatasetLabel, enrichmentTestType, correction, supportedgenomes, annDataSets)
data_from_api = fetch_data_from_api(url)
enrichment_results = filter_enrichment_results(data_from_api, pvalue_threshold=0.05)
enrichment_results.to_csv(os.path.join(path_to_save, f'panther_enrichment_{organismLabel}_{annDatasetLabel}.csv'))



organismLabel = 'human'
annDatasetLabel = 'biological_process'
url = create_pantherdb_url(genes, organismLabel, annDatasetLabel, enrichmentTestType, correction, supportedgenomes, annDataSets)
data_from_api = fetch_data_from_api(url)
enrichment_results = filter_enrichment_results(data_from_api, pvalue_threshold=0.05)
enrichment_results.to_csv(os.path.join(path_to_save, f'panther_enrichment_{organismLabel}_{annDatasetLabel}.csv'))

organismLabel = 'human'
annDatasetLabel = 'cellular_component'
url = create_pantherdb_url(genes, organismLabel, annDatasetLabel, enrichmentTestType, correction, supportedgenomes, annDataSets)
data_from_api = fetch_data_from_api(url)
enrichment_results = filter_enrichment_results(data_from_api, pvalue_threshold=0.05)
enrichment_results.to_csv(os.path.join(path_to_save, f'panther_enrichment_{organismLabel}_{annDatasetLabel}.csv'))



enrichment_results.to_dict()





















