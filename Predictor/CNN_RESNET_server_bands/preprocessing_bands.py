
import os
import glob
from pathlib import Path


image_size=512

imagesPath="../../../data/MetasufacesData/Images/processed128"
imagesDestination="../../../data/MetasufacesData/Images_512_Bands"

folders=glob.glob(imagesPath+"/*.png", recursive = True)
files=[]



# index=0
# for folder in folders:
#     fileName_absolute = os.path.basename(folder) 
#     type_=fileName_absolute.split("_")[0]

#     os.chmod(os.path.abspath(folder), 0o755)
#     os.rename(os.path.abspath(folder), os.path.abspath(imagesDestination+"/"+type_+"/"+fileName_absolute) ) 
#     index+=1
#     print(index)
#     #if index==10000:
#     #   break


import random

testIamgesPath ="../../../data/MetasufacesData/Images_512_Bands/"
Origin ="../../../data/MetasufacesData/Images_512_Bands/box/box_01_freq_reflect_e3744294-aba9-11ee-b4d0-047c16a08772_0-34_40-50.png/"

classesImages=glob.glob(Origin+"/*.png", recursive = True)

index=0
for folder in classesImages:
    fileName_absolute = os.path.basename(folder) 
    print(fileName_absolute)
    os.chmod(os.path.abspath(Origin), 0o755)
    os.rename(os.path.abspath(Origin), os.path.abspath(testIamgesPath+"/"+fileName_absolute) ) 
    #index+=1
    print(index)
    
    #if index==2:
    #  break