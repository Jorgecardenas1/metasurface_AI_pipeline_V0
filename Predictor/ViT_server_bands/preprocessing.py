
import os
import glob
from druidaHFSS.modules import tools

image_size=512
imagesPath="../../../data/MetasufacesData/Images"

folders=glob.glob(imagesPath+"/*/", recursive = True)
files=[]

processed="/processed512/"

print(folders)

for folder in folders:
    
    if folder != imagesPath+processed:
        files=(files+glob.glob(folder+"/*"))


for file in files:
    fileName_absolute = os.path.basename(file) 
    path=os.path.dirname(file)

    #ROI is 
    image_rgb=tools.cropImage( file,image_path=processed,
                              image_name=fileName_absolute,
                              output_path=imagesPath, 
                             resize_dim=(image_size,image_size))
        
