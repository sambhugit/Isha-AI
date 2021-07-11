from flask import Flask, request, jsonify, render_template
import json
import os
import numpy as np
import datetime
from datetime import datetime
import Huggingface_model as hf
import string
import tensorflow
import keras
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import random
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib
from matplotlib import pyplot as plt
import Pydrive as pdd
import cleaners as cl
from PIL import Image
import base64
import io
import re
import time
import Google_drive_downloader 




app = Flask('Isha',template_folder='template')

current_dir = os.path.dirname(os.path.realpath('app.py'))

emotion_var = [0]*8
user_chat_list = []
hugging_face_model_list = []

#first_message = 0

# load chatbot dataset
with open(os.path.join(current_dir,'intents.json'), errors = "ignore") as file:
    data = json.load(file)

# load chatbot model

model_chat = load_model(os.path.join(current_dir,'Model_3.hdf5'))

# load chatbot tokenizer object
with open(os.path.join(current_dir,'tokenizer_emo.pickle'), 'rb') as handle:
    tokenizer = pickle.load(handle)

# load label encoder object
with open(os.path.join(current_dir,'label_encoder_emo.pickle'), 'rb') as enc:
    lbl_encoder = pickle.load(enc)

# parameters
max_len = 20 
max_review_length = 200

# load sentiment analysis model
weight_path = os.path.join(current_dir,'sentiment_weight_file.hdf5')
model_sa = load_model(weight_path)

# load sentiment analysis tokenizer object
with open(os.path.join(current_dir,'tokenizer_sa_new1.pickle'), 'rb') as handle:
    tokenizer_sa = pickle.load(handle)

def process_reddit_comment(strng):
    # remove [NAME] placeholder
    processed_strng = re.sub('\[name]', '', strng)
    # remove reddit symbol 
    processed_strng = re.sub('/r', '', processed_strng)
    return processed_strng


def punct_remover(strng):
    # punctuation marks to be completely removed
    clean_strng = re.sub(r'[?|!|\'|"|#]', r'', strng)
    # punctuation marks to be replaced with space
    clean_strng = re.sub(r'[.|,|)|(|\|/]', r' ', clean_strng)
    # replace multi-space with single space 
    clean_strng = re.sub(r' +', r' ', clean_strng)

    return clean_strng


def sentiment_call(inp):
    
    list_to_be_analysed = []
    list_to_be_analysed = inp
    list_to_be_analysed = cl.remove(list_to_be_analysed)
    result = [0]
    outcome_labels = [0,1]

    #print("inp =",inp,"\n")
    #print("list_to_be_analysed=",list_to_be_analysed,"\n")

    for m in list_to_be_analysed:
        #print("m=",m,"\n")
        s = [m]
        seq = tokenizer_sa.texts_to_sequences(s)
        padded = pad_sequences(seq, maxlen=max_review_length) 
        pred = model_sa.predict(padded)
        result = np.append(result,outcome_labels[np.argmax(pred)])
    return result

def identify_sentiment(sentiment,emotion_var):
    
    index = [0,1]
    sentiment = np.delete(sentiment,index)
    sentiment_scores = sentiment
    value = sum(sentiment)/(len(sentiment))
    if value >= 0.5:
        user_sentiment = "User is Not Depressed"
    else:
      user_sentiment = "User is Depressed"
    
    return user_sentiment, sentiment_scores



@app.route('/result', methods = ['POST'])
def results():

    res = sentiment_call(user_chat_list)
    sentiment = np.array([0])
    sentiment = np.append(sentiment,res)
    user_sentiment, sentiment_scores = identify_sentiment(sentiment,emotion_var)

    sum_emotion_var = sum(emotion_var,0)

    #graph = plt.figure()

    mylabels = [
                  "clam "+str((emotion_var[0]/sum_emotion_var)*100),"regret "+str((emotion_var[1]/sum_emotion_var)*100),
                  "afraid "+str((emotion_var[2]/sum_emotion_var)*100),"confused "+str((emotion_var[3]/sum_emotion_var)*100),
                  "care "+str((emotion_var[4]/sum_emotion_var)*100),"sad "+str((emotion_var[5]/sum_emotion_var)*100),
                  "angry "+str((emotion_var[6]/sum_emotion_var)*100),"happy "+str((emotion_var[7]/sum_emotion_var)*100)
               ]
      
    mycolors = ['r','g','b','c','m','y','#FF7F50','#B8860B','#4CAF50']
     
    fig = plt.figure()
    plt.pie(emotion_var,labels = mylabels, colors = mycolors)
    plt.legend()
    fig.savefig(os.path.join(current_dir,"client_emotions/image.png"))
    im = Image.open(os.path.join(current_dir,"client_emotions/image.png"))
    data_img = io.BytesIO()
    im.save(data_img, "PNG")
    encoded_img_data = base64.b64encode(data_img.getvalue())
    path = os.path.join(current_dir,"client_emotions/image.png")
    os.remove(path)
    with open(os.path.join(current_dir,"client_datax/Client_Data%s.txt" %date_time), 'a') as handle:
         handle.write("\n" + "emotion_var=" + str(emotion_var) +"\n" + "sentiment_scores="+ str(sentiment_scores)+"\n")
    pdd.uploader_func(current_dir)
    #path_data = os.path.join(current_dir,"client_datax")
    #path_data = os.path.join(current_dir,"client_datax\Client_Data%s.txt"%date_time)
    os.remove(os.path.join(current_dir,"client_datax/Client_Data%s.txt"%date_time))
    return render_template('graph.html', graph = encoded_img_data.decode('utf-8') , user_sentiment = user_sentiment )

# Chat function
@app.route('/chat', methods=['POST'])
def chat():

    user_chat = request.form['chat']

    if user_chat.lower() in ["who are you?", "what are you?", "who you are?","are you human?","are you a robot?","who are you", "what are you", 
                             "who you are","are you human","are you a robot","what is you","what is you?"  ]:
        
        reply_bot = np.random.choice(["I,m Isha, your bot assistant", "I'm Isha, an Artificial Intelligent bot","I am your virtual friend"])
        return render_template('indexfinal.html', user_chat = user_chat, reply_bot = reply_bot)

    if user_chat.lower() in ["what is your name", "what should I call you", "whats your name?","your good name please","your good name please?",
                             "your good name, please?","your good name, please","what is your name?","what should I call you?","whats your name","may i know your name","may i know your name?",
                             "how should i address you","how should i address you?","tell me your name?","tell me your name"]:
        
        reply_bot = np.random.choice(["You can call me Isha.", "I'm Isha!", "Just call me as Isha"])
        return render_template('indexfinal.html', user_chat = user_chat, reply_bot = reply_bot)
     
    #message = request.form['message']
    #global first_message = first_message + 1
    user_chat_processed = user_chat
    user_chat_processed = process_reddit_comment(user_chat_processed.lower())
    user_chat_processed = punct_remover(user_chat_processed)


    result = model_chat.predict(pad_sequences(tokenizer.texts_to_sequences([user_chat_processed]),
                                              truncating='post', maxlen=max_len))                                  
    tag = lbl_encoder.inverse_transform([np.argmax(result)])

    global emotion_var
    
    if tag == 'calmness':
        emotion_var[0] = emotion_var[0] + 1
    elif tag == 'regret': 
      emotion_var[1] = emotion_var[1] + 1
    elif tag == 'fear': 
      emotion_var[2] = emotion_var[2] + 1
    elif tag == 'confused':
      emotion_var[3] = emotion_var[3] + 1
    elif tag == 'care':
      emotion_var[4] = emotion_var[4] + 1
    elif tag == 'sad':
      emotion_var[5] = emotion_var[5] + 1
    elif tag == 'angry':
      emotion_var[6] = emotion_var[6] + 1
    elif tag == 'happy':
      emotion_var[7] = emotion_var[7] + 1

    user_chat_list.append(user_chat)
    #if first_message == 2:
    checker = 0
    
    while checker == 0:
      try:
        reply = hf.getuserchat(user_chat, user_chat_list, hugging_face_model_list)
        #print(reply)
        reply_bot = reply['generated_text']
        checker = 1
      except:
        checker = 0
    
    hugging_face_model_list.append(reply_bot)
    #print(reply_bot)



    return render_template('indexfinal.html', user_chat = user_chat, reply_bot = reply_bot)
    
    

# Form function
@app.route('/form', methods=['POST'])
def form():
 
    gender_value = request.form['gender']
    age_value = request.form['age']
    covid_value = request.form['covid']

    dateTimeObj = datetime.now()
    #global date_time 
    date_time_temp = dateTimeObj
    for x in str(dateTimeObj):
        if x == ':':
            date_time_temp = str(dateTimeObj).replace(x,'')

    global date_time
    date_time = date_time_temp

    y = os.path.join(current_dir,"client_datax/Client_Data %s.txt" % date_time)
    #print(y)
    file = open( y, 'w')  
    file.write(gender_value + "\n"  + age_value + "\n" + covid_value)

    file.close()


    return render_template('index.html')




# start function

@app.route('/',methods = ['GET'])
def ping():
    
    return render_template('vertida.html')

# Main

if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0', port = 9696)



