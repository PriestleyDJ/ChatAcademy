import os
import requests
import xml.etree.ElementTree as ET
import argparse
from requests.auth import HTTPBasicAuth

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process some user details.')
parser.add_argument('username', type=str, help='Username for authentication')
parser.add_argument('password', type=str, help='Password for authentication')
parser.add_argument('directory', type=str, help='Directory containing the files')
args = parser.parse_args()

# Define the directory containing the user files
directory = args.directory

# Define directories for saving XML data based on type
publication_dir = os.path.join(directory, 'publications')
grant_dir = os.path.join(directory, 'grants')
journal_dir = os.path.join(directory, 'journals')
activity_dir = os.path.join(directory, 'activities')
relationship_dir = os.path.join(directory, 'relationships')

# Create directories if they don't exist
os.makedirs(publication_dir, exist_ok=True)
os.makedirs(grant_dir, exist_ok=True)
os.makedirs(journal_dir, exist_ok=True)
os.makedirs(activity_dir, exist_ok=True)
os.makedirs(relationship_dir, exist_ok=True)

# Credentials
username = args.username
password = args.password

def extract_user_details(user_id, path):
    user_url = f"https://mypublications.shef.ac.uk:8091/secure-api/v6.13/users/{user_id}"
    
    response = requests.get(user_url, auth=HTTPBasicAuth(username, password))
    
    xml_data = response.text
    # Parse XML
    root = ET.fromstring(xml_data)
    
    ns = {'api': 'http://www.symplectic.co.uk/publications/api'}
    
    # Extract details
    full_name = root.find(".//api:first-name", ns).text + " " + root.find(".//api:last-name", ns).text
    department = root.find(".//api:organisation-defined-data[@field-name='DEPARTMENT']", ns).text
    is_academic = root.find(".//api:is-academic", ns).text == 'true'
    primary_group_descriptor = root.find(".//api:primary-group-descriptor", ns)
    primary_group_descriptor = primary_group_descriptor.text if primary_group_descriptor is not None else "unknown"
    
    title = root.find(".//api:title", ns)
    title = title.text if title is not None else "unknown"
    
    with open(path, "w") as file:
        file.write(f"{full_name} is associated with the {department} department\n")
        if is_academic:
            file.write(f"{full_name} is an academic\n")
        else:
            file.write((f"{full_name} is not an academic\n"))
        file.write(f"{full_name}'s title is {title}\n")
        file.write(f"{full_name}'s primary group descriptor is {primary_group_descriptor}\n")
        
    return full_name

# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".xml"):
        user_id = filename.split('.')[0]
        print(f"Processing: {user_id}")
        
        relationship_path = os.path.join(relationship_dir, f"{user_id}.txt")
        
        user_name = extract_user_details(user_id, relationship_path)
        
        url = f"https://mypublications.shef.ac.uk:8091/secure-api/v6.13/users/{user_id}/relationships"

        # Continue looping as long as there is a URL to call
        while url:
            response = requests.get(url, auth=HTTPBasicAuth(username, password))
            if response.status_code == 200:
                xml_data = response.text
                root = ET.fromstring(xml_data)
                namespaces = {'api': 'http://www.symplectic.co.uk/publications/api'}
                
                # Process each publication
                for publication in root.findall('.//api:result', namespaces=namespaces):
                    relationship = publication.find('.//api:relationship', namespaces=namespaces)
                    related_object = relationship.find('.//api:object', namespaces=namespaces)
                    
                    pub_id = related_object.get('id')
                    pub_type = related_object.get('category')
                    pub_date = related_object.get('created-when')
                    print(f"Publication ID: {pub_id}, Type: {pub_type}, Created When: {pub_date}")

                    # Determine the save path based on the publication type
                    if pub_type.lower() == "publication":
                        save_path = os.path.join(publication_dir, f"{pub_id}.xml")
                        api_endpoint_base = f'https://mypublications.shef.ac.uk:8091/secure-api/v6.13/publications/{pub_id}'
                    elif pub_type.lower() == "grant":
                        save_path = os.path.join(grant_dir, f"{pub_id}.xml")
                        api_endpoint_base = f'https://mypublications.shef.ac.uk:8091/secure-api/v6.13/grants/{pub_id}'
                    elif pub_type.lower() == "journal":
                        save_path = os.path.join(journal_dir, f"{pub_id}.xml")
                        api_endpoint_base = f'https://mypublications.shef.ac.uk:8091/secure-api/v6.13/journals/{pub_id}'
                    elif pub_type.lower() == "activity":
                        save_path = os.path.join(activity_dir, f"{pub_id}.xml")
                        api_endpoint_base = f'https://mypublications.shef.ac.uk:8091/secure-api/v6.13/activities/{pub_id}'
                    else:
                        continue
                    
                    response = requests.get(api_endpoint_base, auth=HTTPBasicAuth(username, password))
                    print(response.status_code)
                    # Check if the call was successful
                    if response.status_code == 200:
                        # The request was successful; we can get the XML data from the response
                        xml_data = response.text
                        
                        # Parse XML
                        root = ET.fromstring(xml_data)
                    
                        # Namespace is required to access elements
                        ns = {'api': 'http://www.symplectic.co.uk/publications/api'}
                    
                        # Save the XML data to the determined path
                        with open(save_path, 'w', encoding='utf-8') as file:
                            file.write(xml_data)
                            print(f"Saved XML to {save_path}")
                        
                        with open(relationship_path, 'a', encoding='utf-8') as file:
                            if pub_type.lower() == "publication":
                                # Extract the publication title
                                title_element = root.find(".//api:field[@name='title']/api:text", ns)
                                title = title_element.text if title_element is not None else "unknown"
                                # Extract the publication type
                                publication_type_element = root.find(".//api:object[@category='publication']", ns)
                                publication_type = publication_type_element.get("type-display-name") if publication_type_element is not None else "unknown"
                                
                                # Print the extracted information
                                file.write(f"{user_name} has a publication titled '{title}' which is a {publication_type}\n")
                        
                            elif pub_type.lower() == "grant":
                                # Extract grant title
                                title_element = root.find(".//api:field[@name='title']/api:text", ns)
                                title = title_element.text if title_element is not None else "unknown"                      
                                # Print the extracted information
                                file.write(f"{user_name} is associated with '{title}' grant\n")
                        
                            elif pub_type.lower() == "journal":
                                # Extract the journal title
                                title_element = root.find(".//api:field[@name='title']/api:text", ns)
                                title = title_element.text if title_element is not None else "unknown"
                                # Extract publisher
                                publisher_element = root.find(".//api:field[@name='publisher']/api:text", ns)
                                publisher = publisher_element.text if publisher_element is not None else "unknown"
                                # Print the extracted information
                                file.write(f"{user_name} has a journal titled '{title}' which is published by {publisher}\n")
                        
                            elif pub_type.lower() == "activity":
                                # Extract the organisation name
                                organisation_element = root.find(".//api:field[@name='c-org']/api:text", ns)
                                organisation_name = organisation_element.text if organisation_element is not None else "unknown"
                                # Extract department name
                                department_element = root.find(".//api:field[@name='department']/api:text", ns)
                                department = department_element.text if department_element is not None else "unknown"
                                # Extract activity description
                                description_element = root.find(".//api:field[@name='description']/api:text", ns)
                                description = description_element.text if description_element is not None else "unknown"
                              
                                # Print the extracted information
                                file.write(f"{user_name} is associated with an activity with description of '{description}' by the organisation {organisation_name} and {department} department\n")
                                               
                # Find the URL for the next page, if it exists
                next_page = root.find('.//api:page[@position="next"]', namespaces=namespaces)
                url = next_page.get('href') if next_page is not None else None
            else:
                print(f"Failed to retrieve data: {response.status_code}")
                break
