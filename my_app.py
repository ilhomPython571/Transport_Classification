import pathlib
# pathlib.PosixPath = pathlib.WindowsPath
import platform
import streamlit as st
from fastai.vision.all import *
import plotly.express as px

plt = platform.system()
if plt == 'Linux':pathlib.WindowsPath=pathlib.PosixPath


# title
st.title('Transportni klassifikatsiya qiluvchi model')

# rasm joylash
file = st.file_uploader('Rasm yuklash ', type=['png', 'jpeg', 'gif', 'svg'])
if file:
    st.image(file)

    # PIL convert
    img = PILImage.create(file)

    # model
    model = load_learner('transport_model.pkl')

    # prediction
    pred, pred_id, probs = model.predict(img)
    st.success(f'Bashorat : {pred}')
    st.info(f'Ehtimollik : {probs[pred_id]*100:.2f}')
    
    # plotting
    fig = px.bar(x=probs, y=model.dls.vocab)
    st.plotly_chart(fig)
    
