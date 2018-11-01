import csv
import re
import numpy as np
import pandas as pd
import sys
import random

#    https://keras.io/getting-started/sequential-model-guide/#examples
#  check this out
#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs
#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs
#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, TimeDistributed, Bidirectional
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing import sequence 
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python import keras
from tensorflow.python.keras.callbacks import ModelCheckpoint


random.seed(100)


#batch_size = 200  #BI
#Dbatch_size = 400  #BI
batch_size = 500
#epoch_size = 1500
epoch_size = 1000
epoch_size = 600
epoch_size = 599   #BIDIRECTIONAL
epoch_size = 399   #BIDIRECTIONAL
epoch_size = 1198  #BI Yosemite
epoch_size = 1111  #BI Yosemite
#epoch_size = 855  #BI best checkpoint is 854
#epoch_size = 99   #BIDIRECTIONAL
#epoch_size = 850  #BI Yosemite
#epoch_size = 999  #BI Yosemite removed a_1_1
#epoch_size = 350
#epoch_size = 1


symbol_list= [ "\\sigma_1_1", "(_1_1",       "\\sum_1_1",   "1_1_1",       "n_1_1",       "2_1_1",       ")_1_1",       "r_1_1",      
 "i_2_1",       "\\theta_1_1", "\\sum_2_bot", "b_1_1",       "c_1_1",       "4_1_1",       "3_1_1",       "d_1_1",      
"a_1_1",     "8_1_1",       "7_1_1",       "4_2_nose",    "y_1_1",       "0_1_1",       "x_2_left",   
 "x_1_1",       "\\sqrt_1_1",  "o_1_1",       "u_1_1",       "\\mu_1_1",    "k_1_1",      "\\lt_1_1",
  "q_1_1",        "\\{_1_1",     "\\}_1_1",     
 "9_1_1",        "\\int_1_1",   "t_2_tail",    "e_1_1",       "x_2_right",       "g_1_1",       "s_1_1",      
 "5_2_hook",    "6_1_1",       "v_1_1",       "5_1_1",       "w_1_1",       "\\gt_1_1",    "\\alpha_1_1",
 "\\beta_1_1",  "\\gamma_1_1", "m_1_1",       "l_1_1",        "\\infty_1_1", "/_1_1"]      

# "\\lim_2_lim",  
# "\\cos_1_cos", 
# "\\cos_2_co", 
# "\\log_2_lo", 
# "\\lim_3_li", 
# "\\log_1_log", 
# "\\tan_3_an", 

#$ "\\{_1_1",     "\\}_1_1",     "]_1_1",    "[_1_1",    
#$
#$[_1_1, ]_1_1, (_1_1, x_2_right, L_1_1, p_2_ear, "\\{_1_1",     "\\}_1_1",
#https://gist.github.com/RyanAkilos/3808c17f79e77c4117de35aa68447045
#confusion matrix generator


#file_ = './abc_head_essence_final_2013.csv'
#data_file = 'abc_train_2011_12_13_new_regul.csv'
#data_file = 'abc_train_2011_12_13_new_regul_clean_x.csv' #after I removed strange x_2_left and x_2_right
#data_file = 'abc_train_2011_12_13_new_regul_clean_x_balanced_2.csv' #undersample 2_1_1 so that the count would be similar to x_2_left
#data_file = 'hijk_output_features.csv'
#data_file = 'abc_manual_cleaned_features_extracted.csv'#hijk_output_features.csv'
data_file = 'abc_features_extracted.csv'


#t = []
#k = []
#l = []
#s = []
#if(True):
def read_training_data( data_file, symbol_list):
    t = []
    k = []
    l = []
    s = []
    #data_file = './hijk_ordered_output_features.csv'
    #file_ = './abc_essence_final_2013.csv'
    tmp_n = [] #number
    tmp_t = [] #trace
    tmp_s = [] #symbol
    verbose = False
    #verbose = True
    with open(data_file, 'r') as f:
        #reader = csv.reader(f, delimiter=',', quotechar = '"', doublequote = True, quoting=csv.QUOTE_NONE)
        reader = csv.reader(f, skipinitialspace=True) # delimiter=',', quotechar = '"', doublequote = True, quoting=csv.QUOTE_NONE)
        #next(reader, None)  #skipping header
        for row in reader:
            #trace_regular =  row[13] #prior to Oct 3, when I modified regulization, I shifted 5 metric in add_trace_bb R function
    #        trace_regular =  row[18]
            tmp_n.append(row[0])
            tmp_t.append(row[1])
            tmp_s.append(row[2])
    empty = pd.DataFrame()
    empty['num'] = pd.Series(tmp_n).values
    empty['trace'] = pd.Series(tmp_t).values
    empty['sym'] = pd.Series(tmp_s).values
    print("====empty =====")
    #print(empty)
    #print(empty.loc[0:2, ['sym']])
    sorted_empty = empty.sort_values(by=['sym'])
    print("==== sorted_empty =====")
    #print(sorted_empty)
    df_count_by_group = sorted_empty.groupby(['sym']).size()
    print("==== df_count_by_group =====")
    #print(df_count_by_group)
#
    stripped_df = sorted_empty.loc[sorted_empty['sym'].isin(symbol_list)] #choose only useful symbols
    #print(stripped_df)
    stripped_count_by_group = stripped_df.groupby(['sym']).size()
    strip_count_df = stripped_count_by_group.to_frame() #convert Series to data frame
    print("==== strip_count_df =====")
    #print(strip_count_df)
    total_df = pd.DataFrame()
    for index, row in strip_count_df.iterrows(): #for val in stripped_count_by_group:
        #print(index)
        copy_num = int(np.ceil(1000/row[0]))
    #    print(" index %s and sample count %d " %(index, copy_num)) #print value only
        current_df = stripped_df.loc[stripped_df['sym'] == index]
        #print( lll )
        if(copy_num > 1):
            increased_df = current_df.append([current_df]*copy_num,ignore_index=True)
        else:
            increased_df = current_df
        increased_df = increased_df.sample(n=1000)
        total_df = total_df.append(increased_df)
   # print(total_df.groupby(['sym']).size())
   # print("==== total_df =====")
   # print(total_df)
    print("==== Kera Version  =====")
    print(keras.__version__)
    print(keras.__file__)
    print("==== total_df shuffling, string to numpy formatting =====")
    total_df_shuffle = total_df.sample(frac=1)
    #t = total_df_shuffle['trace'].tolist()
    #s = total_df_shuffle['num'].tolist()
    #converting str to array for network
    count = 0
    for index, row in total_df_shuffle.iterrows():
        count = count + 1
        #if(count > 1):
        #  sys.exit()
        num = row[0]
        trace = row[1]
        sym = row[2]
        refindout = re.findall(r"[-+]?[0-9]*\.?[0-9]+", trace)
        map_float = np.array( list( map(float, refindout)))
        strokes = np.reshape( map_float , (-1, 15))
        t.append(strokes)
        k.append(sym)
        l.append(len(strokes))
        s.append(num)
        if(count == 1 and False):
          print("======= trace ======")
          print(trace)
          print(type(trace))
          print("======= map_float ======")
          print(map_float)
          print(type(map_float))
          print("======= strokes ======")
          print(strokes)
          print(type(strokes))
          print("======= sym ======")
          print(sym)
          print(type(sym))
          print("======= len ======")
          print(len(strokes))
          print("======= index ======")
          print(index)
          print("======= types ======")
          print(type(t))
          print(type(k))
          print(type(l))
          print(type(s))
          print(len(t))
          print(len(k))
          print(len(l))
          print(len(s))
    return (t,k,l,s)


t,k,l,s = read_training_data( data_file, symbol_list)
#file_ = './abc_essence_final_2013.csv'
#t = [] 
#k = []
#l = []
#s = []
#verbose = False
##verbose = True
#with open(data_file, 'r') as f:
#    #reader = csv.reader(f, delimiter=',', quotechar = '"', doublequote = True, quoting=csv.QUOTE_NONE)
#    reader = csv.reader(f, skipinitialspace=True) # delimiter=',', quotechar = '"', doublequote = True, quoting=csv.QUOTE_NONE)
#    #next(reader, None)  #skipping header
#    for row in reader:
#        #trace_regular =  row[13] #prior to Oct 3, when I modified regulization, I shifted 5 metric in add_trace_bb R function
##        trace_regular =  row[18]
#        trace_regular =  row[1]
#        key =  row[-1]
#
#        #https://www.regular-expressions.info/floatingpoint.html
#        refindout = re.findall(r"[-+]?[0-9]*\.?[0-9]+", trace_regular)
#        map_float = np.array( list( map(float, refindout)))
#        if(False):
#          print(row)
#          print(trace_regular)
#          print(key)
#          print(refindout)
#          print(map_float)
#        strokes = np.reshape( map_float , (-1, 15))
#        if( key in  symbol_list):
#          t.append(strokes)
#          k.append(key)
#          l.append(len(strokes))
#          s.append(row[0]) #sequence
#
#        if(verbose):
#          print(row)
#          print(trace_regular)
#          print(refindout)
#          print( map_float)


t = np.asarray(t)

datasize = len(l)
print( " read %d strokes from %s " % (datasize, data_file ))
#print(l)

def TRACE_print( t ):
  for trace in t:
     print( "len:%d, first %s %s, last %s %s" % (len(trace), trace[0,0], trace[0,1], trace[-1,0],trace[-1,1]))
     print( trace.shape )
     
def zeropad ( t, timestep): # t is batch size of (N, 2) , N is number of points
  batch = len(t)
  #print( " batch %d , timestep %d " % (batch, timestep))
  #print(timestep)
  new_x = np.zeros((batch, timestep, 15))
  for i in range(0,batch):
     #print( len(t[i,]) )
     new_x[i,:len(t[i,])] = t[i,]
     #print(t[i,])
     #print(new_x[i,])
  return(new_x)

def findindex ( lib, keys ):
  idex = []
  #library = lib.tolist()
  for k in keys:
    #print(k)
    idex.append(lib.index(k))
  #print(idex)
  return(idex)


##sequence_length = 100'
##k_list = np.sort(k)
##print(k_list)
#k_set = np.sort(list(set( k )))
#
#
##idx = findindex( k_set, list(["a_1_1", "b_1_1", "z_1_1"]))
##print(idx)
##k_list = list(k_set)
#print( k_set )
#k_count = len(k_set)
#print(k_count)

#current = 0
verbose = False
def train_generator():
  global current
  global datasize
  global t, k, l, s
  global verbose
  #global k_set, k_count
  global symbol_list
  current = 0

  while True:

        begin = current
        end = begin + batch_size
        if( end > datasize): 
            end = end - datasize
            x_train = np.append(t[begin:], t[:end])
            y_train = np.append(k[begin:], k[:end]) #k[begin:] + k[:end]
            l_train = np.append(l[begin:], l[:end])
            s_train = np.append(s[begin:], s[:end])            
        else:
            x_train = t[begin:end]
            y_train = k[begin:end]
            l_train = l[begin:end]
            s_train = s[begin:end]        
        

        current = end
        timestep = int( max(l_train) )

        x_train = zeropad(x_train, timestep) # np.zeros((timestep, 2))
        idx = findindex( symbol_list,  y_train)
        category_y = to_categorical(idx , num_classes = len(symbol_list))
        #TRACE_print( x_train )
        #print( "x_shape ")
        if(verbose):
          print( " begin %d , end %d " % (begin, end))
          print(x_train.shape)
          print( y_train )
          print(category_y)
 
        #print( l_train )
        if(verbose):
          print( s_train )
        #print( "max_time %d" % (timestep) )

##        if(i == 500*100):
##           print(i)
          print(x_train.shape)
##           print(x_train[i,])
##           print(y_train[i])

        yield x_train, category_y

#t_g = train_generator()
#for i in range(0,10):
#    print( " ------------------------------- %d"% i )
#    next(t_g)



first_node = 64
second_node = 32
model = Sequential()
#model.add(LSTM(32, return_sequences=True, input_shape=(None, 2)))
#model.add(LSTM(8, return_sequences=False)) 

#model.add(LSTM(first_node, return_sequences=True, input_shape=(None, 2)))
#model.add(LSTM(16, return_sequences=False)) 
#model.add(LSTM(second_node, return_sequences=False)) 

model.add(Bidirectional(LSTM(first_node, return_sequences=True), input_shape=(None, 15)))
model.add(Bidirectional(LSTM(second_node, return_sequences=False))) 



#model.add((Dense(3, activation='sigmoid')))  
model.add((Dense(len(symbol_list), activation='softmax')))  

#The sigmoid activation is outputing values between 0 and 1 independently from one another.
#If you want probabilities outputs that sums up to 1, use the softmax activation on your last layer, it will normalize the output to sum up to 1. 

#model.add((Dense(2, activation='softmax')))  

	#timedistributed requires all sequences (return_sequences = True). 
	#https://stackoverflow.com/questions/43034960/many-to-one-and-many-to-many-lstm-examples-in-keras

print(model.summary(90))

model.compile(loss='categorical_crossentropy',
                optimizer='adam',  metrics=['mse', 'accuracy'])
#    https://github.com/keras-team/keras/issues/2548
#    evaluate() will return the list of metrics that the model was compiled with. 
#    So if your compile metrics list include 'accuracy', then this should still work.

 

#http://parneetk.github.io/blog/neural-networks-in-keras/
#https://keras.io/callbacks/
#earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=1, mode='auto')

filepath="weights.checkpoint.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#earlystop = EarlyStopping(monitor='loss', min_delta=0.0001, patience=5, verbose=1, mode='auto') #restore_best_weight is not supporting in my version
#callbacks_list = [earlystop]
#model.fit_generator(train_generator(), steps_per_epoch=30, epochs=10, verbose=1)
num_batch_in_epoch = round(datasize / batch_size) + 1
print( "number of batches in an epoch %d" % (num_batch_in_epoch))
#model.fit_generator(train_generator(), steps_per_epoch=num_batch_in_epoch, epochs=30, verbose=1)
#history_callback = model.fit_generator(train_generator(), steps_per_epoch=30, epochs=epoch_size, verbose=1)
history_callback = model.fit_generator(train_generator(), steps_per_epoch=30, epochs=epoch_size, verbose=1, callbacks=callbacks_list)
#history_callback = model.fit_generator(train_generator(), steps_per_epoch= num_batch_in_epoch, epochs=epoch_size, verbose=1
#history_callback = model.fit_generator(train_generator(), steps_per_epoch= num_batch_in_epoch, epochs=epoch_size, verbose=1)



print(history_callback.history.keys())

fi_stem = "./model_sym%d_batch%d_epoch%d_1st%d_2nd%d" % (len(symbol_list), batch_size, epoch_size, first_node, second_node)
save_file = "%s.h5"%(fi_stem)
model.save(save_file)

model_json = model.to_json()
js_file = "./deploy_model_sym%d_batch%d_epoch%d_1st%d_2nd%d.json" % (len(symbol_list), batch_size, epoch_size, first_node, second_node)
with open( js_file,  "w") as json_file:
  json_file.write(model_json)
weight_file = "./deploy_model_sym%d_batch%d_epoch%d_1st%d_2nd%d_weight.h5" % (len(symbol_list), batch_size, epoch_size, first_node, second_node)
model.save_weights(weight_file)

loss_file = "%s_loss.txt"%(fi_stem)
acc_file = "%s_acc.txt"%(fi_stem)
mse_file = "%s_loss.txt"%(fi_stem)

loss_history = history_callback.history['loss']
numpy_loss_history = np.array(loss_history)
np.savetxt(loss_file, numpy_loss_history, delimiter=",")

#acc_history = history_callback.history['acc']
#numpy_acc_history = np.array(acc_history)
#np.savetxt(acc_file, numpy_acc_history, delimiter=",")

#mse_history = history_callback.history['mean_squared_error']
#numpy_mse_history = np.array(mse_history)
#np.savetxt(mse_file, numpy_mse_history, delimiter=",")

