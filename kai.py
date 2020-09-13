import io
import os
from google.cloud import vision
from google.cloud.vision import types
import pandas as pd
import numpy as np

# ----memo----
# joy 1~5
# interested 1, not 0
# often_laugh 1, not 0
# knowledge 1, ない   0
# time,joy,interested,often_laugh,knowledgeの形に

def readfiles(filename):
    ary = []
    for item in filename:
        interested = []
        with open('data/{}.txt'.format(item)) as f:
            interested = f.readlines()
            interested = [int(s.strip()) for s in interested]
        ary += interested
    return np.array(ary)

def pleasant_estimation():
    filename = ["fukushima","houjin","neishi","oikawa","ueda"]
    often_laugh = [0,1,1,1,0]
    knowledge = [0,1,1,1,1]
    list_df = pd.DataFrame(columns=["name","time", "joy", "interested", "often_laugh", "knowledge"])
    n = 20
    interested = readfiles(filename)
    for index, item in enumerate(filename):
        for i in range(n):
            # Instantiates a client
            client = vision.ImageAnnotatorClient()
            # The name of the image file to annotate
            file_name = os.path.abspath('{}_img/{}.png'.format(item, str(i)))
            # Loads the image into memory
            with io.open(file_name, 'rb') as image_file:
                content = image_file.read()
            image = types.Image(content=content)
            # Performs label detection on the image file
            response = client.face_detection(image=image)
            faces = response.face_annotations
            for face in faces:
                tmp = pd.Series([item, i, face.joy_likelihood, interested[n*index + i], often_laugh[index], knowledge[index]], index=list_df.columns)
                list_df = list_df.append(tmp,ignore_index=True)
    print(list_df.dtypes)
    return(list_df)

# ファイル読み込み部分の関数化