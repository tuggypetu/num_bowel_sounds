
# streamlit run app.py

import streamlit as st

# from pydub import AudioSegment
from datetime import datetime

import os

import tensorflow
import librosa
from librosa import load, stft, amplitude_to_db, display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow.keras

# st.write("Hello ,let's learn how to build a streamlit app together")
# st.title ("this is the app title")
# st.header("this is the markdown")
# st.markdown("this is the header")
st.subheader("Upload audio file to get number of bowel sounds.")
# st.caption("this is the caption")
# st.code("x=2021")
# st.latex(r''' a+a r^1+a r^2+a r^3 ''')

audio_file = st.file_uploader("Upload audio file", type=['mp3', 'wav', 'ogg'])



def predict_img(audio_fil):
    #import model
    model = tensorflow.keras.models.load_model('my_model')

    #convert X
    X = get_x(audio_fil).reshape(-1, 200, 15, 4)
    #predict
    prediction = model.predict(X).reshape(-1,200).ravel()
    #one-hot-result
    result = one_hot_X(prediction)
    return f"    {len(result)} number of bowel sounds in {audio_file.name}"


def get_x(filename):
    # audio_data, sr = librosa.load(filename)
    # short time fourier transform
    # stft = librosa.stft(audio_data, n_fft=478, hop_length=221, window='hann', center=True)
    stft1 = stft(filename, n_fft=478, hop_length=221, window='hann', center=True)
    Xdb = amplitude_to_db(stft1).T
    # standardisation
    mean = Xdb.mean()
    deviation = Xdb.std()
    standard_value = (Xdb - mean) / deviation
    X1 = []
    for i in range(len(Xdb)):
        array = []
        result = []
        for j in range(len(standard_value[0])):
            array.append(standard_value[i][j])
            if j % 4 == 0:
                result.append(sum(array) / len(array))
                array = []
        X1.append(result)
    X = np.abs(np.array(X1).reshape(200, 15, 4))
    return X


def one_hot_X(prediction):
    for i in range(len(prediction)):
        if prediction[i] < 0.5:
            prediction[i] = 0
        else:
            prediction[i] = 1
    # print(prediction)
    return [x for x in prediction if x ==1]


def save_audio(file):
    if file.size > 4000000:
        return 1
    # if not os.path.exists("audio"):
    #     os.makedirs("audio")
    folder = "audio"
    datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    # clear the folder to avoid storage overload
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    try:
        with open("log0.txt", "a") as f:
            f.write(f"{file.name} - {file.size} - {datetoday};\n")
    except:
        pass

    with open(os.path.join(folder, file.name), "wb") as f:
        f.write(file.getbuffer())
    return 0


if audio_file is not None:
    # audio_bytes = audio_file.read()
    # audio_file = st.audio(audio_bytes, format='audio/wav')
    # target = os.path.join('temp/' + audio_file.name)  # location of image present in temp directory
    # audio_file.export(target)

    sav_path = f"images/spec_mine.png"
    path = os.path.join("audio", audio_file.name)
    if_save_audio = save_audio(audio_file)

    gg = st.audio(audio_file, format='audio/wav', start_time=0)

    y, sr = load(path, duration=2)
    X = stft(y, n_fft=441, hop_length=110, window='hann')
    Xdb = amplitude_to_db(np.abs(X), ref=np.max)

    fig, ax = plt.subplots(figsize=(72, 72))
    img = display.specshow(Xdb, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title=audio_file.name)
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.savefig(sav_path)
    plt.close(fig)

    # Opens a image in RGB mode
    im = Image.open(sav_path)

    # Setting the points for cropped image
    left = 890.5
    top = 5457.5
    right = 5464.6
    bottom = 6511.5

    # Cropped image of above dimension
    # (It will not change original image)
    im1 = im.crop((left, top, right, bottom))

    im1.save(sav_path)

    print(im.size, ',', im1.size)
    im.close()
    im1.close()

    spec_image = st.image(sav_path)

    audio_data, s_rate = librosa.load(path, duration=2)
    st.header(predict_img(audio_data))


# streamlit run app.py
