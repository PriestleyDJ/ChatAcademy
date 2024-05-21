# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:01:35 2024

@author: Eva Szwagrzak
"""

import requests
import xml.etree.ElementTree as ET
import argparse
from requests.auth import HTTPBasicAuth

# Parse command line arguments
parser = argparse.ArgumentParser(description='Fetch details from API based on category.')
parser.add_argument('username', type=str, help='Username for authentication')
parser.add_argument('password', type=str, help='Password for authentication')
parser.add_argument('category', type=str, help='Category to fetch details for')
args = parser.parse_args()

# Credentials
username = args.username
password = args.password

# Category id to fetch user files for 
category = args.category

def call_api(api_endpoint, category):
    # Make a GET request to the API endpoint
    response = requests.get(api_endpoint, auth=HTTPBasicAuth(username, password))
    
    # Check if the call was successful
    if response.status_code == 200:
        # The request was successful; we can get the XML data from the response
        xml_data = response.text
        print("XML Data received from the API:")
        
        # Parse the XML response
        root = ET.fromstring(xml_data)
        
        # Namespace mapping to handle namespaced tags
        namespaces = {'api': 'http://www.symplectic.co.uk/publications/api'}
        
        # Find all category object elements
        category_ids = [obj.get('id') for obj in root.findall(f'.//api:object[@category="{category}"]', namespaces)]
        next_page = root.find('.//api:page[@position="next"]', namespaces)
        
        pagination = root.find('.//api:pagination', namespaces)
        results_count = int(pagination.get('results-count'))
        items_per_page = int(pagination.get('items-per-page'))
            
        # Calculate the number of pages
        total_pages = (results_count + items_per_page - 1) // items_per_page
        
        
        next_page_href = next_page.get('href') if next_page is not None else None
        
        print("category IDs:", category_ids)
    
        return category_ids, next_page_href, total_pages
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return [], None, 0

# Base API endpoint
api_endpoint_base = f'https://mypublications.shef.ac.uk:8091/secure-api/v6.13/{category}s/'

# Function to fetch details for a specific category
def fetch_category_details(category_id):
    api_endpoint = f"{api_endpoint_base}{category_id}"
    response = requests.get(api_endpoint, auth=HTTPBasicAuth(username, password))
    if response.status_code == 200:
        xml_data = response.text
        # Parse the XML data
        root = ET.fromstring(xml_data)
        
        # Find the 'is-current-staff' element and check if it's 'true'
        is_staff = root.find(".//api:is-current-staff", {'api': 'http://www.symplectic.co.uk/publications/api'})
        if is_staff is not None and is_staff.text == 'true':
            # Save the data to a file only if the user is current staff
            with open(f'{category_id}.xml', 'w', encoding='utf-8') as file:
                file.write(xml_data)
            print(f"Data for current staff user {category_id} saved successfully.")
        else:
            print(f"User {category_id} is not current staff or tag missing.")
    else:
        print(f"Failed to retrieve data for user {category_id}: {response.status_code}")

# API endpoint to start with
api_endpoint = f"https://mypublications.shef.ac.uk:8091/secure-api/v6.13/users?groups=120&amp;group-membership=explicit"

category_ids, next_page, total_pages = call_api(api_endpoint, category)
current_page = 1

while next_page and current_page <= total_pages:
    print(f"Processing page {current_page}/{total_pages}")
    for category_id in category_ids:
        fetch_category_details(category_id)
    
    if next_page:
        category_ids, next_page, _ = call_api(next_page, category)
        current_page += 1
