#Includes twilio setup, dropbox, detection and all

import os
from twilio.rest import Client
from flask import Flask
from twilio.twiml.voice_response import VoiceResponse
import dropbox
import io
from urllib.request import urlopen
import librosa
import json
import pydub
from dropbox.exceptions import AuthError
import dropbox
import requests
#from tensorflow.keras.preprocessing import image
import keras_preprocessing.image as image
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer as at
from transformers import AutoModelForSequenceClassification as asc
import torch
import pickle
from scipy.special import softmax
import pickle
from flask import Flask, request, Response
import os
import speech_recognition as srrr
import json
import pandas as pd
import csv
from csv import writer

def dropbox_connect():
    """Create a connection to Dropbox."""
    try:
        dbx = dropbox.Dropbox('sl.BOF1e6BTtI2Txkgcu1L_mB9o5JhmvoCyOQSk9y_VUZMDjaGMjLHzofSuumuAW0dOWU4yxI1JRKW3xNN5ZI6ekGrkAy-BG5qAZUjUqD7uVBQeZqYKv2yAv_uer0sAVVLWx8l5-YAh')
    except AuthError as e:
        print('Error connecting to Dropbox with access token: ' + str(e))
    return dbx
app = Flask(__name__)
@app.route("/record", methods=['GET', 'POST'])
def record():
    """Returns TwiML which prompts the caller to record a message"""
    # Start our TwiML response
    response = VoiceResponse()

    # Use <Say> to give the caller some instructions
    response.say("Hello.I am from team 405 found. It's dial hundred. What's your problem.")

    # Use <Record> to record the caller's message
    response.record(max_length=10, transcribe=True, recording_status_callback='/recording/callback')
    # End the call with <Hangup>
    #response.hangup()

    return str(response)
@app.route('/recording/callback', methods=['POST'])
def upload_recording():
    try:
        recording_url = request.form['RecordingUrl']
        recording_sid = request.form['RecordingSid']
        call_sid = request.form['CallSid']
        client = Client('AC331ff148de6cd8952d107b4c97064f72', 'f6125755a58e7d45a3c77ebad049f6fe')
        calls = client.calls.list(limit=1)
        r = requests.get('https://api.twilio.com/2010-04-01/Accounts/AC331ff148de6cd8952d107b4c97064f72/Calls/' + call_sid + '.json',
            auth=('AC331ff148de6cd8952d107b4c97064f72', 'f6125755a58e7d45a3c77ebad049f6fe'))
        resp = r.content
        my_json = resp.decode('utf8').replace("'", '"')
        new = json.loads(my_json)
        with open("log_file.json", "w") as f:
            f.write(json.dumps(new))
        from_no = new["from"]
        dropbox_client = dropbox.Dropbox('sl.BOF1e6BTtI2Txkgcu1L_mB9o5JhmvoCyOQSk9y_VUZMDjaGMjLHzofSuumuAW0dOWU4yxI1JRKW3xNN5ZI6ekGrkAy-BG5qAZUjUqD7uVBQeZqYKv2yAv_uer0sAVVLWx8l5-YAh')
        upload_path = f"/sihrecordings/{recording_sid}.wav"
        with requests.get(recording_url, stream=True) as r:
            dropbox_client.files_upload(r.raw.read(), upload_path)
        try:
                dbx = dropbox_connect()
                shared_link_metadata = dbx.sharing_create_shared_link_with_settings(upload_path)
                shared_link = shared_link_metadata.url
                shared_link=shared_link.replace('?dl=0', '?dl=1')
                print(shared_link)
        except dropbox.exceptions.ApiError as exception:
                if exception.error.is_shared_link_already_exists():
                    shared_link_metadata = dbx.sharing_list_shared_links(upload_path)
                    shared_link = shared_link_metadata.links[0].url
                    shared_link = shared_link.replace('?dl=0', '?dl=1')
                    print(shared_link)
        print('yes')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        print('no')
        wav = io.BytesIO()
        with urlopen(shared_link) as r:
            print(r)
            r.seek = lambda *args: None  # allow pydub to call seek(0)
            print(r.seek)
            sound=pydub.AudioSegment.from_file(r)
            sound.export(wav, "wav")
        wav.seek(0)
        y, sr = librosa.load(wav)
        #y, sr = librosa.load(shared_link)
        rrr = srrr.Recognizer()
        # open the file
        with open('C:\\Users\\mithu\\PycharmProjects\\twilio\\newa.wav', 'wb') as f:
            metadata, result = dbx.files_download(path=upload_path)
            f = f.write(result.content)
        try:
            with srrr.AudioFile('C:\\Users\\mithu\\PycharmProjects\\twilio\\newa.wav') as source:
                audio_data = rrr.record(source)
                text = rrr.recognize_google(audio_data)
                print(text)
                q = 2
        except Exception as eee:
            print('Robo voice/Blank call  detected.Rejecting the call!')
            q = 1
            #os.exit(1)
        print(sr)

        # try:
        #     del df["Unnamed: 0"]
        # except:
        #     pass
        df = pd.DataFrame(columns=["Phone Number", "Text", "No. of Fake", "No. of Legit", "Credibility Score"])

        if from_no not in df.values:
            # list=[from_no,text,0,0,100]
            # temp = pd.DataFrame([list])
            # temp.columns = ["Phone Number", "Text",	"No. of Fake",	"No. of Legit",	"Credibility Score"]
            # lst = [df, temp]
            # result = pd.concat(lst)
            # result.reset_index(drop=True)
            # result.to_csv("file.csv")
            df.loc[len(df)] = [from_no, text, 0, 0, 100]

        ms = librosa.feature.melspectrogram(y, sr=sr)
        log_ms = librosa.power_to_db(ms, ref=np.max)
        librosa.display.specshow(log_ms, sr=sr)
        fig.savefig("new1.png")
        plt.close(fig)
        images_new = []
        new_img = images_new.append(image.img_to_array(image.load_img("new1.png", target_size=(256, 256, 3))))
        x_new = []
        x_new += images_new
        x_new_norm = np.array(x_new) / 255
        nsamples, nx, ny, nrgb = x_new_norm.shape
        x_new2 = x_new_norm.reshape((nsamples, nx * ny * nrgb))
        print(x_new2)
        loaded_model = pickle.load(open("rf_img.sav", 'rb'))
        a = loaded_model.predict(x_new2)
        hp = (loaded_model.predict_proba(x_new2)[0][1] * 100)
        rp = (loaded_model.predict_proba(x_new2)[0][0] * 100)
        bp = 0
        if (q == 1):
            os._exit(1)
        if (q == 2):
            if (a == 1):
                print('Human voice detected! Forwarding your call to dial 100')
                tokenizer = at.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
                model = asc.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
                encoded_input = tokenizer(text, return_tensors='pt')
                output = model(**encoded_input)
                pred = output[0][0].detach().numpy()
                pred = softmax(pred)
                print("Negative Statement: {}".format(pred[0] * 100))
                print("Neutral Statement: {}".format(pred[1] * 100))
                print("Positive Statement: {}".format(pred[2] * 100))
                print('Percentage of Human: {}'.format(loaded_model.predict_proba(x_new2)[0][1] * 100))
                print('Percentage of Robo: {} '.format(loaded_model.predict_proba(x_new2)[0][0] * 100))
                inp = input("Give input whether to mark as fake or legit. For Fake, enter 'F' and for Legit enter 'L'")
                if inp.lower() == "f":
                    row_no = df[df['Phone Number'] == from_no].index.values
                    new = df.iloc[row_no]
                    new["No. of Fake"] = new["No. of Fake"] + 1
                    new["Credibility Score"] = new["Credibility Score"] - 5
                    df.iloc[row_no] = new
                    print(df)
                elif inp.lower() == "l":
                    row_no = df[df['Phone Number'] == from_no].index.values
                    new = df.iloc[row_no]
                    new["No. of Legit"] = new["No. of Legit"] + 1
                    df.iloc[row_no] = new
                    print(df)
            else:
                print('Robo voice detected! Declining the call  {} '.format(
                    loaded_model.predict_proba(x_new2)[0][0] * 100))
        with open("params.txt", "w") as file:
            file.write(str(loaded_model.predict_proba(x_new2)[0][1] * 100)+' ')
            file.write(str(loaded_model.predict_proba(x_new2)[0][0] * 100)+' ')
            file.write(df["Credibility Score"].to_string())
            file.close()
      #  os.system('app.py 1')
    except Exception as e:
        print(e)
    return Response(), 200

if __name__ == "__main__":
    app.run(debug=True)
