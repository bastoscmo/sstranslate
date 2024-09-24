#!/usr/bin/env python3

#################################################
# Siesta Structure Translator -sstranslator     #
# version 1.0                                   #
# UnB - 2024/08/21                              #
#                                               #
# Supported Files:  * XYZ,                      #
#                   * POSCAR (VASP),            #
#                   * Siesta STRUCT_OUT         #
#                                               #
# Dr. Carlos M. O. Bastos                       #
#################################################



# Import libraries
import numpy as np
import sys


# Dictionary with all periodic elements table
element = {
    "H": 1,     # Hidrogênio
    "He": 2,    # Hélio
    "Li": 3,    # Lítio
    "Be": 4,    # Berílio
    "B": 5,     # Boro
    "C": 6,     # Carbono
    "N": 7,     # Nitrogênio
    "O": 8,     # Oxigênio
    "F": 9,     # Flúor
    "Ne": 10,   # Neônio
    "Na": 11,   # Sódio
    "Mg": 12,   # Magnésio
    "Al": 13,   # Alumínio
    "Si": 14,   # Silício
    "P": 15,    # Fósforo
    "S": 16,    # Enxofre
    "Cl": 17,   # Cloro
    "Ar": 18,   # Argônio
    "K": 19,    # Potássio
    "Ca": 20,   # Cálcio
    "Sc": 21,   # Escândio
    "Ti": 22,   # Titânio
    "V": 23,    # Vanádio
    "Cr": 24,   # Cromo
    "Mn": 25,   # Manganês
    "Fe": 26,   # Ferro
    "Co": 27,   # Cobalto
    "Ni": 28,   # Níquel
    "Cu": 29,   # Cobre
    "Zn": 30,   # Zinco
    "Ga": 31,   # Gálio
    "Ge": 32,   # Germanio
    "As": 33,   # Arsênio
    "Se": 34,   # Selênio
    "Br": 35,   # Bromo
    "Kr": 36,   # Kriptônio
    "Rb": 37,   # Rubídio
    "Sr": 38,   # Estrôncio
    "Y": 39,    # Ítrio
    "Zr": 40,   # Zircônio
    "Nb": 41,   # Nióbio
    "Mo": 42,   # Molibdênio
    "Tc": 43,   # Tecnécio
    "Ru": 44,   # Rutênio
    "Rh": 45,   # Ródio
    "Pd": 46,   # Paládio
    "Ag": 47,   # Prata
    "Cd": 48,   # Cádmio
    "In": 49,   # Índio
    "Sn": 50,   # Estanho
    "Sb": 51,   # Antimônio
    "Te": 52,   # Telúrio
    "I": 53,    # Iodo
    "Xe": 54,   # Xenônio
    "Cs": 55,   # Césio
    "Ba": 56,   # Bário
    "La": 57,   # Lantânio
    "Ce": 58,   # Cério
    "Pr": 59,   # Praseodímio
    "Nd": 60,   # Neodímio
    "Pm": 61,   # Promécio
    "Sm": 62,   # Samário
    "Eu": 63,   # Európio
    "Gd": 64,   # Gadolínio
    "Tb": 65,   # Térbio
    "Dy": 66,   # Disprósio
    "Ho": 67,   # Holmium
    "Er": 68,   # Érbio
    "Tm": 69,   # Túlio
    "Yb": 70,   # Itérbio
    "Lu": 71,   # Lutécio
    "Hf": 72,   # Háfnio
    "Ta": 73,   # Tântalo
    "W": 74,    # Wolframio
    "Re": 75,   # Rênio
    "Os": 76,   # Ósmio
    "Ir": 77,   # Irídio
    "Pt": 78,   # Platina
    "Au": 79,   # Ouro
    "Hg": 80,   # Mercúrio
    "Tl": 81,   # Tálio
    "Pb": 82,   # Chumbo
    "Bi": 83,   # Bismuto
    "Po": 84,   # Polônio
    "At": 85,   # Astato
    "Rn": 86,   # Radônio
    "Fr": 87,   # Frâncio
    "Ra": 88,   # Radônio
    "Ac": 89,   # Actínio
    "Th": 90,   # Tório
    "Pa": 91,   # Protactínio
    "U": 92,    # Urânio
    "Np": 93,   # Netúnio
    "Pu": 94,   # Plutônio
    "Am": 95,   # Amerício
    "Cm": 96,   # Curió
    "Bk": 97,   # Berquélio
    "Cf": 98,   # Califórnio
    "Es": 99,   # Einstênio
    "Fm": 100,  # Férmio
    "Md": 101,  # Mendelevio
    "No": 102,  # Nobelio
    "Lr": 103,  # Laurêncio
    "Rf": 104,  # Rutherfórdio
    "Db": 105,  # Dúbnio
    "Sg": 106,  # Seabórgio
    "Bh": 107,  # Bóhrio
    "Hs": 108,  # Hassio
    "Mt": 109,  # Meitnério
    "Ds": 110,  # Darmstádio
    "Rg": 111,  # Roentgênio
    "Cn": 112,  # Copernício
    "Nh": 113,  # Nihônio
    "Fl": 114,  # Fleróvio
    "Mc": 115,  # Moscóvio
    "Lv": 116,  # Livermório
    "Ts": 117,  # Tenesso
    "Og": 118   # Oganessônio
}

# Dictionary with all atomic numbers table
atomicnumber = {str(v): k for k, v in element.items()}


###### Functions Definition

# This Function define the number of atoms types for xyz file
def typeofatoms(dataxyz,element):
  atoms=[]
  j=1
  listatoms=[]
  for i in range(2,len(dataxyz),1):
    if dataxyz[i][0] not in atoms:
        atoms.append(dataxyz[i][0])
        listatoms.append([j,element[dataxyz[i][0]],dataxyz[i][0]])
        j=j+1
  return listatoms

# This Function define the number of atoms types for vasp file
def getatomsandvectors(datavasp):
  latticeparameter=datavasp[1]
  vectors=datavasp[2:5]
  typevectors=datavasp[7]
  getatomsvasp=[]
  for i in range(len(datavasp[5])):
    getatomsvasp.append([i+1,element[datavasp[5][i]], datavasp[5][i],datavasp[6][i]])
  return typevectors,latticeparameter,vectors,getatomsvasp


# function get the atoms and vector from siesta structure_out file
def getatomsandvectorssiesta(datasiesta):
  latticeps=datasiesta[:3]
  getatomsps=[]
  for i in range(4,len(datasiesta),1):
    getatomsps.append(datasiesta[i])
  nossiesta=[]
  atoms=[]
  for i in range(len(getatomsps)):
    if getatomsps[i][1] not in atoms:
        atoms.append(getatomsps[i][1])
        nossiesta.append([str(i+1),getatomsps[i][1],atomicnumber[getatomsps[i][1]]])
  return latticeps,getatomsps,nossiesta



# Function to Read a xyz file
def readfilexyz(filename):
  filexyz=open(filename,"r")
  dataxyz=[]
  for lines in filexyz:
    lines=lines.replace("\n", "")
    lines=lines.split(" ")
    noblank=[]
    for item in lines:
      if item.strip():
          noblank.append(item)
    dataxyz.append(noblank)
  return dataxyz

# Function to Write a fdf file from xyz
def writefilefdfxyz(dataxyz,listatoms):
  outfile=[]
  outfile.append('# automatic create translation from XYZ file using sstranslate')
  outfile.append(" ")
  outfile.append("NumberOfSpecies    "+str(len(listatoms)) )
  outfile.append("NumberofAtoms      "+str(len(dataxyz)-2))
  outfile.append(" ")
  outfile.append("%block ChemicalSpeciesLabel ")
  for i in range(len(listatoms)):
    outfile.append("   "+str(listatoms[i][0])+"   "+str(listatoms[i][1])+"   "+str(listatoms[i][2]))
  outfile.append("%endblock ChemicalSpeciesLabel")
  outfile.append(" ")
  outfile.append("LatticeConstant 1 Ang")
  outfile.append(" ")
  outfile.append("AtomicCoordinatesFormat  Ang")
  outfile.append(" ")
  outfile.append("%block LatticeVectors")
  outfile.append(" ATENTTION: THE XYZ DONT CONTAIN THIS INFORMATION")
  outfile.append("%endblock LatticeVectors")
  outfile.append(" ")
  outfile.append("%block AtomicCoordinatesAndAtomicSpecies")
  for i in range(2,len(dataxyz),1):
   for j in range(len(listatoms)):
      if dataxyz[i][0] in listatoms[j]:
        outfile.append("   "+str(dataxyz[i][1])+"   "+str(dataxyz[i][2])+"   "+str(dataxyz[i][3])+"   "+str(listatoms[j][0]))
  outfile.append("%endblock AtomicCoordinatesAndAtomicSpecies")
  return outfile

# Function to Write a fdf file from vasp
def writefilefdfvasp(datavasp,typevectors,latticeparameter,vectors,getatomsvasp):
  outfile=[]
  outfile.append('# automatic create translation from POSCAR (VASP) file using sstranslate')
  outfile.append(" ")
  outfile.append("NumberOfSpecies    "+str(len(getatomsvasp)) )
  outfile.append("NumberofAtoms      "+str(len(datavasp)-8))
  outfile.append(" ")
  outfile.append("%block ChemicalSpeciesLabel ")
  for i in range(len(getatomsvasp)):
    outfile.append("   "+str(getatomsvasp[i][0])+"   "+str(getatomsvasp[i][1])+"   "+str(getatomsvasp[i][2]))
  outfile.append("%endblock ChemicalSpeciesLabel")
  outfile.append(" ")
  outfile.append("LatticeConstant "+str(latticeparameter[0])+"  Ang")
  outfile.append(" ")
  if typevectors[0]=="Direct":
    outfile.append("AtomicCoordinatesFormat Fractional")
  if typevectors[0]=="Cartesian":
    outfile.append("AtomicCoordinatesFormat  Ang")
  outfile.append(" ")
  outfile.append("%block LatticeVectors")
  for i in range(len(vectors)):
    outfile.append("   "+str(vectors[i][0])+"   "+str(vectors[i][1])+"   "+str(vectors[i][2]))
  outfile.append("%endblock LatticeVectors")
  outfile.append(" ")
  outfile.append("%block AtomicCoordinatesAndAtomicSpecies")
  k=8
  for i in range(len(getatomsvasp)):
    for j in range(int(getatomsvasp[i][3])):
      outfile.append("   "+str(datavasp[k][0])+"   "+str(datavasp[k][1])+"   "+str(datavasp[k][2])+"   "+str(getatomsvasp[i][0]))
      k=k+1
  outfile.append("%endblock AtomicCoordinatesAndAtomicSpecies")
  return outfile

# Function to Write a fdf file from siesta stuct_out
def writefilefdfsiesta(vectorsiesta,getatomssiesta,nosiesta):
  outfile=[]
  outfile.append('# automatic create translation from SIESTA STRUCT_OUT file using sstranslate')
  outfile.append(" ")
  outfile.append("NumberOfSpecies    "+str(len(nossiesta)))
  outfile.append("NumberofAtoms      "+str(len(getatomssiesta)))
  outfile.append(" ")
  outfile.append("%block ChemicalSpeciesLabel ")
  for i in range(len(nosiesta)):
    outfile.append("   "+str(nosiesta[i][0])+"   "+str(nosiesta[i][1])+"   "+str(nosiesta[i][2]))
  outfile.append("%endblock ChemicalSpeciesLabel")
  outfile.append(" ")
  outfile.append("LatticeConstant 1.0  Ang")
  outfile.append(" ")
  outfile.append("AtomicCoordinatesFormat Fractional")
  outfile.append(" ")
  outfile.append("%block LatticeVectors")
  for i in range(len(vectorsiesta)):
    outfile.append("   "+str(vectorsiesta[i][0])+"   "+str(vectorsiesta[i][1])+"   "+str(vectorsiesta[i][2]))
  outfile.append("%endblock LatticeVectors")
  outfile.append(" ")
  outfile.append("%block AtomicCoordinatesAndAtomicSpecies")
  for i in range(len(getatomssiesta)):
    for j in range(len(nossiesta)):
      if getatomssiesta[i][1] in nossiesta[j]:
        outfile.append("   "+str(getatomssiesta[i][2])+"   "+str(getatomssiesta[i][3])+"   "+str(getatomssiesta[i][4])+"   "+str(nossiesta[j][0]))
  outfile.append("%endblock AtomicCoordinatesAndAtomicSpecies")
  return outfile


###### Program Start
# Choose 1 for XYZ file
# Choose 2 for POSCAR file
# Choose 3 for Siesta STRUCT_OUT file

filename=sys.argv[1]

if sys.argv[3]=="xyz":
    option=1
elif sys.argv[3]=="poscar":
    option=2
elif sys.argv[3]=="siesta":
    option=3
else:
    print ("----------------------------------------\n")
    print ("!!!!!file format not define!!!!!")
    print ("use the flag --help for more information")
    print ("\n---------------------------------------\n\n\n")


#option=input("Choose 1 for XYZ file or 2 for POSCAR file: ")
#option=int(option)
#if option!=1 and option!=2 and option!=3:
#  print("Option not valid")
#  exit()

if option==1:
  dataxyz=readfilexyz(filename) # Read the file
  listatoms=typeofatoms(dataxyz,element) # Get the number of atoms types
  outfile=writefilefdfxyz(dataxyz,listatoms) # Create the file fdf from xyz
  np.savetxt("structure.fdf",outfile,fmt='%s') # Save the file fdf from xyz
  print("File fdf created")

if option==2:
  datavasp=readfilexyz(filename) # Read the file
  typevectors,latticeparameter,vectors,getatomsvasp=getatomsandvectors(datavasp) # Get the number of atoms types
  outfilevasp=writefilefdfvasp(datavasp,typevectors,latticeparameter,vectors,getatomsvasp)
  np.savetxt("structure.fdf",outfilevasp,fmt='%s') # Save the file fdf from vasp
  print("File fdf created")

if option==3:
  datasiesta=readfilexyz(filename) # Read the file
  vectorsiesta,getatomssiesta,nossiesta=getatomsandvectorssiesta(datasiesta)
  outfilesiesta=writefilefdfsiesta(vectorsiesta,getatomssiesta,nossiesta)
  np.savetxt("structure.fdf",outfilesiesta,fmt='%s') # Save the file fdf from vasp
  print("File fdf created")




