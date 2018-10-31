

#https://www.youtube.com/watch?v=f6Bf3gl4hWY
#Sraji 
#train an image classifier in Keras using a Tensorflow backend, then serve it to the browser 
#using a super simple Flask backend. We can then deploy this flask app to google cloud using a few commands. Woot! 


#record trace and play back
#http://ramkulkarni.com/blog/record-and-playback-drawing-in-html5-canvas/


import csv
import re
import numpy as np


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, TimeDistributed
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing import sequence 
from tensorflow.python.keras.models import load_model


#model = load_model('./me_ep1000_var61_acc90.h5') #./math_equation.h5', custom_objects = { '0':'a', '1':'b', '2': 'c', '3':'d'})
#model = load_model('./model_sym61_batch500_epoch1000_1st64_2nd32.h5')
#model = load_model('./model_sym20_batch500_epoch600_1st64_2nd32.h5')
model = load_model('./model_sym61_batch200_epoch1199_1st64_2nd32.h5')


model_json = model.to_json()
with open( "deploy_model_sym61_epoch1199.json", "w") as json_file:
  json_file.write(model_json)
model.save_weights("deploy_model_sym61_epoch1199.h5")   

print(model)

#https://stackoverflow.com/questions/42621864/difference-between-keras-model-save-and-model-save-weights
#
#save() saves the weights and the model structure to a single HDF5 file. I believe it also includes things like the optimizer state. Then you can use that HDF5 file with load() to reconstruct the whole model, including weights. save_weights() only saves the weights to HDF5 and nothing else. You need extra code to reconstruct the model from a JSON file.



