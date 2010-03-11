# -*- coding: utf-8 -*-

__author__ = "Joshua Ryan Smith (jrsmith@cmu.edu)"
__version__ = ""
__date__ = ""
__copyright__ = "Copyright (c) 2010 Joshua Ryan Smith"
__license__ = "GPL"

import re
from numpy import *

class StaibDat(dict):
  """
  StaibDat, child of dict.

  The StaibDat class imports data from a Staib AES or XPS .dat file created by 
  the winspectro software, and makes this data available like a python 
  dictionary. The class tests the data file and the data itself before it 
  returns an object. See the README file for more information about the 
  assumptions made regarding the structure of these files. Since there is no 
  published standard of the structure of the Staib .dat file format, the 
  parsing done by this class will be as general as possible.
  
  There are several ways to access data in a StaibDat object. Since the 
  winspectro .dat files resemble the key-value pairs of a dictionary, the user
  can directly access the data pulled in from the file simply via the key in 
  the data file. There are some rules about the names of the keys.
  
  There are several data that all StaibDat objects have (units in brackets):
    filename: The name of the file from which the data in the object came.
    fileText: A list with the full text of the data file. Each line of the file
      is a new list item.
    KE [eV]: A numpy array containing the kinetic energy value of the electrons.
    BE [eV]: A numpy array containing the binding energy of the electrons 
      calculated using the value of the source energy. Note that this array will
      still be calculated for AES data, but will equal KE since the source 
      energy is zero.
    C1 [count]: A numpy array containing the number of counts for a particular
      energy for channel 1.
    C2 [count]: A numpy array containing the number of counts for a particular
      energy for channel 2.
      
  Generally, the user will probably find it easier to work with the KE, BE, etc.
  data as opposed to the dictionary data pulled from the file itself.
  """

  def __init__(self,filename):
    """
    Instantiation of StaibDat object.

    A StaibDat object is instantiated with a string referring to a .dat file 
    containing AES or XPS data generated by winspectro.
    """

    # The StaibDat object should know where its data came from.
    self["filename"] = filename
    
    # Pull in the data from the file and close the file.
    datFile = open(filename,"r")
    self["fileText"] = datFile.readlines()
    datFile.close()
    
    # Verify the data file has the correct structure.
    self.verifystructure()
    
    # Step through the lines and match them.
    for line in self["fileText"]:
      self.parseline(line)
      
    # Verify that the metadata and data in the file agree.
    self.verifydata()
      
    # Generate the user-friendly KE, C1, etc. numpy arrays.
    self.userfriendify()
    
  def verifystructure(self):
    """
    Verify that the imported text data has the proper structure.
    """
    
    lineTypeList = []
    
    for line in self["fileText"]:
      lineTypeList.append(self.verifyline(line))

    if "other" in lineTypeList:
      raise FormatError
    elif lineTypeList.count("datalabels") != 1:
      raise FormatError
    
    compressedList = []
    
    for lineType in lineTypeList:
      if len(compressedList) == 0:
        compressedList.append(lineType)
      elif lineType != compressedList[-1]:
        compressedList.append(lineType)
        
    if compressedList != ["metadata","reserved","datalabels","data"]:
      raise FormatError

  def verifyline(self,line):
    """
    Returns a string indicating what section of the file the line comes from.
    """
    datRE = re.compile("\s+(\d+)\s+(\d+)\s+(\-*\d+)")
    datLabelsRE = re.compile("\s+(Basis\[mV\])\s+(Channel_1)\s+(Channel_2)")
    
    if re.search(r":    ",line):
      return "metadata"
    elif line.strip() == "reserved":
      return "reserved"
    elif datLabelsRE.search(line):      
      return "datalabels"
    elif datRE.search(line):      
      return "data"
    else:
      return "other"
      
  def verifydata(self):
    """
    Compares the values in the metadata section to those in the data section.
    """
    
    # Data Points should equal the number of data points.
    if self["DataPoints"] != len(self["Basis"]["value"]):
      raise FormatError
    
    # The last value of energy should equal Stopenergy.
    if round(self["Stopenergy"]["value"],2) != round(self["Basis"]["value"][-1]/1000,2):
      raise FormatError
       
    # The first value of energy should equal Startenergy.
    if round(self["Startenergy"]["value"],2) != round(self["Basis"]["value"][0]/1000,2):
      raise FormatError
    
    # The difference between each Basis value should be consistent.
    basisList = self["Basis"]["value"][:]
    
    diffList = []
    val = basisList.pop()
    while len(basisList):
      bottomVal = basisList.pop()
      diffList.append(round((val-bottomVal)/1000,2))
      val = bottomVal
      
    if diffList.count(diffList[0]) != len(diffList):
      raise FormatError
    
    # The difference between each Basis value should equal Stepwidth.
    if diffList[0] != round(self["Stepwidth"]["value"],2):
      raise FormatError
       
  def parseline(self,line):
    """
    Determine if line is metadata, reserved, data, or other. Act accordingly.
    """

    # Set up the regular expression necessary to handle the data section of the
    # file.
    # !!!Note: the following re isn't as general as it could be. I could 
    # search for one or more instances of whitespace and not-whitespace.
    datRE = re.compile("\s+(Basis\[mV\]|\d+)" +\
                       "\s+(Channel_1|\d+)" +\
                       "\s+(Channel_2|\-*\d+)")
   
    # metadata
    if re.search(r":    ",line):
      self.metadata(line)

    # reserved
    elif line.strip() == "reserved":
      # Do nothing.
      pass

    # data
    elif datRE.search(line):      
      self.data(line,datRE)
     
  def metadata(self,line):
    """
    Set the metadata as key:value pairs in the StaibDat object.

    In the metadata part of the file, metadata keys and values are separated
    by a colon followed by four whitespace characters. The metadata method 
    splits the line up along that delimiter, prepares the key part of the text
    so that it is a legitimate key string, and then sets the key:value pairs
    in the StaibDat object.

    Some metadata has either implicit or explicit units. The metadata method
    includes these units as part of the value of the key:value pair.
    """

    # Split the key and value and remove/compress the whitespace from the key.
    [key,value] = line.split(":    ")
    # Strip, compress whitespace out of key string.
    key = re.sub("\s+","",key)
    # Strip preceeding and trailing whitespace.
    value = value.strip()

    # Deal with keys that have explicit units.
    if key[-1] == "]":
      # Break off the explicit unit.
      unit = key[-2]
      key = key[0:-3]
      self[key] = {"value":float(value),"unit":unit}
        
    # Deal with keys that have implicit units.
    elif key == "SourceEnergy":
      self[key] = {"value":float(value),"unit":"V"}
    elif key == "Stepwidth":
      self[key] = {"value":float(value),"unit":"V"}
    elif key == "DwellTime":
      self[key] = {"value":float(value),"unit":"ms"}
    elif key == "RetraceTime":
      self[key] = {"value":float(value),"unit":"ms"}

    # Deal with keys that have no units. Coerce the value to int if possible.
    else:
      try:
        value = int(value)
      except ValueError:
        pass
      self[key] = value
   
  def data(self,line,datRE):
    """
    Set the data in the StaibDat object.

    In the data part of the file, the data is arranged in whitespace-separated
    columns. The first line of the data is the labels for the columns, and each
    subsequent line contains the actual data. In the resulting StaibDat object,
    each column of data will have its own key:value pair; the key being the 
    formatted column label, and the value being another dictionary. The value
    dictionary has two keys: unit and value. The unit element is a string 
    containing the unit of the data. The value element is an array containing 
    the values in the column.
    """

    parsedData = datRE.match(line)
    
    if parsedData.group(1) == "Basis[mV]":
      self["Basis"] = {"value":[],"unit":"V"}
      self["Channel_1"] = {"value":[],"unit":"count"}
      self["Channel_2"] = {"value":[],"unit":"count"}
    else:
      self["Basis"]["value"].append(float(parsedData.group(1)))
      self["Channel_1"]["value"].append(float(parsedData.group(2)))
      self["Channel_2"]["value"].append(float(parsedData.group(3)))

  def userfriendify(self):
    """
    Generate the numpy arrays for KE, BE (if available), C1, and C2.
    
    According to conversations with Staib, the analyzer has an internal bias
    and therefore we don't have to compensate for the analyzer work function.
    """
    
    self["KE"] = array(self["Basis"]["value"])/1000
    self["BE"] = self["KE"] - self["SourceEnergy"]["value"]
    self["C1"] = array(self["Channel_1"]["value"])
    self["C2"] = array(self["Channel_2"]["value"])
    
    
class FormatError(Exception):
  """
  """
  pass
  