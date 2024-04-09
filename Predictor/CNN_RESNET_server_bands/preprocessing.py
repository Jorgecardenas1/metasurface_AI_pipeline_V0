
import os
import glob
from druidaHFSS.modules import tools

image_size=512
imagesPath="../../../data/MetasufacesData/Images"

folders=glob.glob(imagesPath+"/*/", recursive = True)
files=[]

processed="/processed128/"

bands = [
    [30,40],[40,50],[50,60],[60,70],[70,80],[80,90]
]

print(folders)

for folder in folders:
    
    if folder != imagesPath+processed:
        files=(files+glob.glob(folder+"/*"))


for file in files:
    for band in bands:

        fileName_absolute = os.path.basename(file) 
        newname=fileName_absolute.split(".")[0]+"_"+str(band[0])+"-"+str(band[1])+".png"
        path=os.path.dirname(file)

        #ROI is 
        image_rgb=tools.cropImage( file,image_path=processed,
                                image_name=newname,
                                output_path=imagesPath, 
                                resize_dim=(image_size,image_size))
        
