from bs4 import BeautifulSoup
from llama_index.core import Document


def readRole(root):
  isStudent = False
  isAcademic = False
  try:
    if(root.find("api:is-student").text == "true"):
      isStudent = True
  except AttributeError:
    isStudent = False

  try:
    if(root.find("api:is-academic").text == "true"):
      isAcademic = True
  except AttributeError:
    isAcademic = False

  if(isStudent == True):
    if(isAcademic == False):
      return "student"
    else:
      return "student and academic"
  else:
    if(isAcademic == True):
      return "academic"
    else:
      return "neither student nor academic"

def readCurrent(root):
  isCurrent = False
  try:
    if(root.find("api:is-current-staff").text == "true"):
      isCurrent = True
  except AttributeError:
    isCurrent = False

  return str(isCurrent)

def readEmail(root):
  email = ""
  try:
    email = root.find("api:email-address").text
  except AttributeError:
    email = "N/A"

  return email

def readDepartment(root):
  department = ""
  try:
    department = root.find("api:primary-group-descriptor").text
  except AttributeError:
    department = "N/A"

  return department

def readArrivalDate(root):
  ArrivalDate = ""
  try:
    ArrivalDate = root.find("api:arrive-date").text
  except AttributeError:
    ArrivalDate = "N/A"

  return ArrivalDate

def readUsername(root):
  username = ""
  try:
    username = root.find("api:object", {"category": "user", "type":"person"})["username"]
  except TypeError:
    username = "N/A"

  return username

def readID(root):
  id_ = ""
  try:
    id_ = root.find("api:object", {"category": "user", "type":"person"})["id"]
  except TypeError:
    id_ = "N/A"
  return id_

def authorReader(autFiles):
  authorDocs = []
  for path in autFiles:
    with open(path, 'r') as f:
      data = f.read()
    root = BeautifulSoup(data, "xml")
    title = root.find("api:title").text
    initials = root.find("api:initials").text
    first_name = root.find("api:first-name").text
    last_name = root.find("api:last-name").text
    role = readRole(root)
    current_member = readCurrent(root)
    email = readEmail(root)
    department = readDepartment(root)
    arrival_date = readArrivalDate(root)
    username = readUsername(root)
    id_ = readID(root)
    doc = Document(
      text = title + " " + first_name + " " + last_name,
      metadata={
          "title": title,
          "initials": initials,
          "first name": first_name,
          "last_name": last_name,
          "role": role,
          "current member": current_member,
          "email": email,
          "department": department,
          "arrival_date": arrival_date,
          "username": username,
          "ID": id_,
          "type" : "person"
          }
    )
    authorDocs.append(doc)
  return authorDocs