import calendar
from bs4 import BeautifulSoup
from llama_index.core import Document


def abstractReader(root):
  abstract = ""
  try:
      abstract = root.find("api:field", {"name": "abstract", "type":"text", "display-name":"Abstract"}).text
  except AttributeError:
     abstract = "N/A"
  return abstract

def authorFromPubReader(root):
  authorsData = root.find("api:field", {"name": "authors", "type":"person-list", "display-name":"Authors"})
  authorData = authorsData.find_all("api:person")
  authorList = []
  for author in authorData:
    initials = author.find("api:initials").text
    try:
      first_name = author.find("api:first-names").text
    except AttributeError:
      first_name = initials
    last_name = author.find("api:last-name").text
    authorDict = {"initials": initials, "first name": first_name, "last name": last_name}
    authorList.append(authorDict)
  return authorList

def pubDateReader(root):
  data = root.find("api:field", {"name": "publication-date", "type":"date", "display-name":"Publication date"})
  day = None
  month = None
  year = None
  try:
    year = data.find("api:year").text
  except AttributeError:
    year = "N/A"
  try:
    month = calendar.month_name[int(data.find("api:month").text)]
  except AttributeError:
    month = "N/A"
  try:
    day = data.find("api:day").text
  except AttributeError:
    day = "N/A"

  return day, month, year

def doiReader(root):
  try:
     doi = root.find("api:field", {"name": "doi", "type":"text", "display-name":"DOI"}).find("api:text").text
  except AttributeError:
     doi= "N/A"
  return doi

def keywordsReader(root):
  try:
     keywords = []
     keywordsXML = root.find("api:keywords")
     keywordsList = keywordsXML.find_all("api:keyword")
     for key in keywordsList:
        keywords.append(key.text)
  except AttributeError:
     keywords = "N/A"
  return keywords

def pubJournalReader(root):
  try:
     journal = root.find("api:field", {"name": "journal", "type":"text", "display-name":"Journal"}).find("api:text").text
  except AttributeError:
     journal = "N/A"
  return journal

def pubDateReader(root):
  data = root.find("api:field", {"name": "publication-date", "type":"date", "display-name":"Publication date"})
  day = None
  month = None
  year = None
  try:
    year = data.find("api:year").text
  except AttributeError:
    year = "N/A"
  try:
    month = calendar.month_name[int(data.find("api:month").text)]
  except AttributeError:
    month = "N/A"
  try:
    day = data.find("api:day").text
  except AttributeError:
    day = "N/A"

  return day, month, year

def doiReader(root):
  try:
     doi = root.find("api:field", {"name": "doi", "type":"text", "display-name":"DOI"}).find("api:text").text
  except AttributeError:
     doi= "N/A"
  return doi

def keywordsReader(root):
  try:
     keywords = root.find("api:field", {"name": "keywords", "type":"keyword-list", "display-name":"Keywords"}).find_all("api:keyword").text
  except AttributeError:
     keywords = "N/A"
  return keywords

def pubJournalReader(root):
  try:
     journal = root.find("api:field", {"name": "journal", "type":"text", "display-name":"Journal"}).find("api:text").text
  except AttributeError:
     journal = "N/A"
  return journal

def publicationReader(pubFiles):
  publicationDocs = []
  for path in pubFiles:
    with open(path, 'r') as f:
      data = f.read()
    root = BeautifulSoup(data, "xml")
    abstract = abstractReader(root)
    authorList = authorFromPubReader(root)
    title = root.find("api:field", {"name": "title", "type":"text", "display-name":"Title"}).text
    keywords = keywordsReader(root)
    date = pubDateReader(root)
    doi = doiReader(root)
    journal = pubJournalReader(root)
    authorsStr = ""
    keywordStr = ""
    if authorList != "N/A":
      for author in authorList:
        authorsStr = authorsStr + f", {author['first name']} {author['last name']}"
      start, _, end = authorsStr[1:].rpartition(", ")
      authorsStr = " The authors of this paper are" + start + " and " + end+ "."
    if keywords != "N/A":
      for keyword in keywords:
        keywordStr = keywordStr + f", {keyword}"
      start, _, end = keywordStr[1:].rpartition(", ")
      keywordStr = " The keywords associated with this paper are " + start + " and " + end + "."
    doc = Document(
      text = f"{title} \n {abstract} \n The authors are {authorsStr} \n This publication was published on {date[0]}/{date[1]}/{date[2]}.\n{authorsStr}\n{keywordStr}",
      metadata={
            "authors": authorList,
            "title": title,
            "keywords": keywords,
            "day published": date[0],
            "month published": date[1],
            "year published": date[2],
            "doi": doi,
            "journal published in": journal
            }
    )
    publicationDocs.append(doc)
  return publicationDocs