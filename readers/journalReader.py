import calendar
from bs4 import BeautifulSoup
from llama_index.core import Document

def licenceReader(root):
  licence = ""
  try:
     licence = root.find("api:field", {"name": "cc-licence", "type":"text"}).text.strip()
  except AttributeError:
     licence = "N/A"
  return licence

def issnReader(root):
  root = root.find("api:field", {"name": "issns", "type":"issn-list", "display-name":"ISSNs"})
  issnData = root.find_all("api:issn")
  issnList = []
  for issn in issnData:
    try:
      issnList.append(issn.text)
    except AttributeError:
      issnList = "N/A"
  return issnList

def endDateReader(root):
  present = True
  endYear = ""
  endMonth = ""
  try:
    root = root.find("api:field", {"name": "end-date", "type":"date", "display-name":"End date"})
  except AttributeError:
    endYear = "N/A"
    endMonth = "N/A"
    present = False

  if(present):
    try:
      endYear = int(root.find("api:year").text)
    except AttributeError:
      endYear = "N/A"
    try:
      endMonth = int(root.find("api:month").text)
    except AttributeError:
      endMonth = "N/A"
  return endMonth, endYear

def journalStartDateReader(root):
  startYear = ""
  try:
    startYear = root.find("api:field", {"name": "start-year-oa", "type":"date", "display-name":"Start year Open Access"}).text.strip()
  except AttributeError:
    startYear = "N/A"
  return startYear

def pubFeeReader(root):
  pubFee = ""
  try:
    pubFee = root.find("api:field", {"name": "publication-fee", "type":"text"}).text.strip()
  except AttributeError:
    pubFee = "N/A"
  return pubFee

def publisherReader(root):
  publisher = ""
  try:
    publisher = root.find("api:field", {"name": "publisher", "type":"text"}).text.strip()
  except AttributeError:
    publisher = "N/A"
  return publisher

def urlReader(root):
  statusRef = ""
  try:
    statusRef = root.find("api:field", {"name": "submission-info-url", "type":"text"}).text
  except AttributeError:
    statusRef = "N/A"
  return statusRef

def journalTitleReader(root):
  title = ""
  try:
    title = root.find("api:field", {"name": "title", "type":"text"}).text
  except AttributeError:
    title = "N/A"
  return title

def journalReader(files):
  journalDocs = []
  for path in files:
    with open(path, 'r') as f:
      data = f.read()
    root = BeautifulSoup(data, "xml")
    title = journalTitleReader(root)
    licence = licenceReader(root)
    issns = issnReader(root)
    pubFee = pubFeeReader(root)
    publisher = publisherReader(root)
    startDate = journalStartDateReader(root)
    url = urlReader(root)
    doc = Document(
      text = title,
      metadata={
            "licence": licence,
            "issns": issns,
            "publication fee": pubFee,
            "publisher": publisher,
            "start date": startDate,
            "url": url,
            "type": "journal"
            }
    )
    journalDocs.append(doc)
  return journalDocs