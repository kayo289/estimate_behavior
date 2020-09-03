import io
import os
# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types
import pandas as pd

# ----memo----
# joy 1~5
# interested 1, not 0
# often_laugh 1, not 0
# knowledge 1, ない   0
# time,joy,interested,often_laugh,knowledgeの形に

def init():
    filename = ["fukushima","houjin","neishi","oikawa","ueda"]
    often_laugh = [0,1,1,1,0]
    knowledge = [0,1,1,1,1]
    list_df = pd.DataFrame(columns=["name","time", "joy", "interested", "often_laugh", "knowledge"])

    for index, item in enumerate(filename):
        # for name in range(5):
        with open('data/{}.txt'.format(item)) as f:
            interested = f.readlines()
            interested = [int(s.strip()) for s in interested]
        for i in range(10):
            # Instantiates a client
            client = vision.ImageAnnotatorClient()
            likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                            'LIKELY', 'VERY_LIKELY')
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
                # print('{}.png: {}'.format(str(i),face.joy_likelihood))
                tmp = pd.Series([item, i, face.joy_likelihood, interested[i], often_laugh[index], knowledge[index]], index=list_df.columns)
                list_df = list_df.append(tmp,ignore_index=True)
    return(list_df)
