import pandas as pd
import os
import shutil
import random
FILE_PATH="metadata.csv"
IMAGES_PATH="images"

df=pd.read_csv(FILE_PATH)
print(df.shape)

df.head()

TARGET_DIR="Dataset/Covid"

if not os.path.exists(TARGET_DIR):
    os.mkdir(TARGET_DIR)
    print("Covid folder created")

cnt=0

for(i,row) in df.iterrows():
    if row["finding"]=="COVID-19" and row["view"]=="PA":
        filename=row["filename"]
        image_path=os.path.join(IMAGES_PATH,filename)
        image_copy_path=os.path.join(TARGET_DIR,filename)
        shutil.copy2(image_path,image_copy_path)
        print("Moving image", cnt)
        cnt+=1
print(cnt)

KAGGLE_FILE_PATH="chest_xray/train/NORMAL"
TARGET_NORMAL_DIR="Dataset/Normal"
image_names=os.listdir(KAGGLE_FILE_PATH)
for i in range(141):
    image_name= image_names[i]
    image_path=os.path.join(KAGGLE_FILE_PATH,image_name)

    target_path=os.path.join(TARGET_NORMAL_DIR,image_name)

    shutil.copy2(image_path,target_path)
    print("Copying")
