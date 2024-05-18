import calendar
from bs4 import BeautifulSoup
from llama_index.core import Document

def amountReader(root):
  amount = ""
  try:
     amount = root.find("api:field", {"name": "amount", "type":"money"}).text
     amount = int("".join(filter(str.isdigit, amount)))
  except AttributeError:
     amount = "N/A"
  return amount

def currencyReader(root):
  currency = ""
  try:
    currency = root.find("api:money")["iso-currency"]
  except TypeError:
    currency = "N/A"
  return currency

def endDateReader(root):
  present = True
  endYear = ""
  endMonth = ""
  root = root.find("api:field", {"name": "end-date", "type":"date", "display-name":"End date"})
  try:
    endYear = int(root.find("api:year").text)
  except AttributeError:
    endYear = "N/A"
  except TypeError:
    endYear = "N/A"
  try:
    endMonth = int(root.find("api:month").text)
  except AttributeError:
    endMonth = "N/A"
  except TypeError:
    endMonth = "N/A"
  return endMonth, endYear

def startDateReader(root):
  present = True
  startYear = ""
  startMonth = ""
  try:
    root = root.find("api:field", {"name": "start-date", "type":"date", "display-name":"Start date"})
  except TypeError:
    startYear = "N/A"
    startMonth = "N/A"
    present = False

  if(present):
    try:
      startYear = int(root.find("api:year").text)
    except AttributeError:
      startYear = "N/A"
    try:
      startMonth = int(root.find("api:month").text)
    except AttributeError:
      startMonth = "N/A"
  return startMonth, startYear

def funderReader(root):
  funder = ""
  try:
    funder = root.find("api:field", {"name": "funder-name", "type":"text"}).text.strip()
  except AttributeError:
    funder = "N/A"
  return funder

def funderReferenceReader(root):
  funderRef = ""
  try:
    funderRef = root.find("api:field", {"name": "funder-reference", "type":"text"}).text.strip()
  except AttributeError:
    funderRef = "N/A"
  return funderRef

def statusReader(root):
  statusRef = ""
  try:
    statusRef = root.find("api:field", {"name": "status", "type":"text"}).text.strip().lower()
  except AttributeError:
    statusRef = "N/A"
  return statusRef

def titleReader(root):
  title = ""
  try:
    title = root.find("api:field", {"name": "title", "type":"text"}).text
  except AttributeError:
    title = "N/A"
  return title

def grantReader(files):
  grantDocs = []
  for path in files:
    with open(path, 'r') as f:
      data = f.read()
    root = BeautifulSoup(data, "xml")
    amount = amountReader(root)
    currency = currencyReader(root)
    endDate = endDateReader(root)
    funder = funderReader(root)
    funderRef = funderReferenceReader(root)
    startDate = startDateReader(root)
    status = statusReader(root)
    title = titleReader(root)
    completeStr = ""
    startStr= ""
    endStr= ""
    if startDate[0] != "N/A" and startDate[1] != "N/A":
      startStr=f" The grant started on {startDate[0]}/{startDate[1]}. "
    if endDate[0] != "N/A" and endDate[1] != "N/A":
      endStr=f"The grant ended on {endDate[0]}/{endDate[1]}. "
    if status != "N/A":
      completeStr=f"The status of the grant is {status}."
    doc = Document(
      text = (f"The grant {title} is funded by {funder}. It is valued at {str(amount)} {currency}." + startStr + endStr + completeStr).replace("\n",""),
      metadata={
          "name": title,
          "amount": amount,
          "currency": currency,
          "end month": endDate[0],
          "end year": endDate[1],
          "start month": startDate[0],
          "start year": startDate[1],
          "funder name": funder,
          "funder reference": funderRef,
          "status": status,
          "type": "grant"
          }
    )
    grantDocs.append(doc)
  return grantDocs