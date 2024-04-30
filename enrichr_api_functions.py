# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 11:07:05 2024

@author: owysocky
"""

import json
import requests


ENRICHR_URL = 'https://maayanlab.cloud/Enrichr/enrich'
query_string = '?userListId=%s&backgroundType=%s'
user_list_id = 363320
gene_set_library = 'KEGG_2015'
response = requests.get(
    ENRICHR_URL + query_string % (user_list_id, gene_set_library)
 )
if not response.ok:
    raise Exception('Error fetching enrichment results')

data = json.loads(response.text)



import requests

base_url = "https://maayanlab.cloud/speedrichr"

genes = [
    'PHF14', 'RBM3', 'MSL1', 'PHF21A', 'ARL10', 'INSR', 'JADE2', 'P2RX7',
    'LINC00662', 'CCDC101', 'PPM1B', 'KANSL1L', 'CRYZL1', 'ANAPC16', 'TMCC1',
    'CDH8', 'RBM11', 'CNPY2', 'HSPA1L', 'CUL2', 'PLBD2', 'LARP7', 'TECPR2', 
    'ZNF302', 'CUX1', 'MOB2', 'CYTH2', 'SEC22C', 'EIF4E3', 'ROBO2',
    'ADAMTS9-AS2', 'CXXC1', 'LINC01314', 'ATF7', 'ATP5F1'
]

description = "sample gene set with background"

res = requests.post(
    base_url+'/api/addList',
    files=dict(
      list=(None, '\n'.join(genes)),
      description=(None, description),
    )
  )
if res.ok:
	userlist_response = res.json()
	print(userlist_response)
    
a = res.json()
