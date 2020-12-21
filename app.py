import streamlit as st 
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def main():

    html1 = """
        <div style="background-color:#cf003e;border-radius:10px;border-style:solid;border-color:black;padding:10px;">
        <h2 style="color:white;text-align:center;">News Headlines Sarcasm Detection</h2>
        </div>
        """
    st.markdown(html1,unsafe_allow_html=True)
    st.write('')
    st.sidebar.title('Machine Learning Project')
    nav = st.sidebar.radio('',['Home','About',"Technologies Used","Dataset"])
    if nav == 'Home':
        st.header('Enter the news headline:')
        sentence = st.text_area('','')
        sentence = [sentence]
        if st.button('Submit'):
            with open("file.txt", 'r', encoding="utf8") as f:
                text = [line.rstrip('\n') for line in f]
        
            new_model = tf.keras.models.load_model('model.h5', compile = False)
            tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
            tokenizer.fit_on_texts(text)
            sequences = tokenizer.texts_to_sequences(sentence)
            padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
            df = pd.DataFrame(new_model.predict(padded),columns=['Value'])

            if(df['Value'][0]>=0.75):
                st.error('Sarcastic')
            elif((df['Value'][0]<0.75)&(df['Value'][0]>=0.6)):
                st.warning('Slightly Sarcastic')
            elif((df['Value'][0]<0.6)&(df['Value'][0]>0.4)):
                st.info('Neutral')
            else:
                st.success('Not Sarcastic')

        else:
            st.subheader('Sarcastic News Headlines Examples: ')
            st.write('-> Miracle cure kills fifth patient.')
            st.write('-> Cows lose their jobs as milk prices drop.')
            st.write('-> Genuine happiness now seen only on game shows.')
        
    if nav == 'About':
        st.subheader('This project is made to implement natural language processing in tensorflow.')
        st.subheader('In this project, I have implemented the Recurrent Neural Network(RNN) with Long Short Term Memory(LSTM) networks in order to detect sarcasm in the news headlines.') 
        st.header('Long short-term memory (LSTM)')
        st.write('LSTM networks are an extension of recurrent neural networks (RNNs) mainly introduced to handle situations where RNNs fail.')
        st.write('Talking about RNN, it is a network that works on the present input by taking into consideration the previous output (feedback) and storing in its memory for a short period of time (short-term memory).')
        st.write('But, there are drawbacks to RNNs:- ')
        html4 = """
        <div>
        <ul>
            <li>It fails to store information for a longer period of time. At times, a reference to certain information stored quite a long time ago is required to predict the current output. But RNNs are absolutely incapable of handling such “long-term dependencies”.</li>
            <li>There is no finer control over which part of the context needs to be carried forward and how much of the past needs to be ‘forgotten’. </li>
            <li>Other issues with RNNs are exploding and vanishing gradients which occur during the training process of a network through backtracking. </li>
        </ul>
        </div>
        """
        st.markdown(html4,unsafe_allow_html=True)
        st.write('Thus, Long Short-Term Memory (LSTM) was brought into the picture. It has been so designed that the vanishing gradient problem is almost completely removed, while the training model is left unaltered.')
        st.write('Long-time lags in certain problems are bridged using LSTMs where they also handle noise, distributed representations, and continuous values.')
        st.write('The complexity to update each weight is also reduced to O(1) with LSTMs.')
        st.write('Also, LSTMs provide us with a large range of parameters such as learning rates, and input and output biases. Hence, no need for fine adjustments.')

    if nav == 'Technologies Used':
        st.write('')
        st.header("Technologies Used")
        html5 = """
        <div>
        <ul>
            <li>Python</li>
            <li>Matplotlib for visualizations</li>
            <li>TensorFlow </li>
            <li>Keras </li>
            <li>Long short-term memory (LSTM)</li>
            <li>Jupyter Notebook</li>
        </ul>
        </div>
        """
        st.markdown(html5,unsafe_allow_html=True)

    if nav == 'Dataset':
        st.write('')
        html2 = """
        <div style='font-size:20px; font-weight: 500;'>
        I have taken real world data from the following link,<br>
        <a href="https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection" style='color:#e60067; text-decoration:none;'>News Headlines Dataset</a>
        </div>
        """
        st.markdown(html2,unsafe_allow_html=True)
        st.subheader('Each record consists of three attributes:')
        html3 = """
        <div>
        <ul>
            <li>is_sarcastic: 1 if the record is sarcastic otherwise 0</li>
            <li>headline: the headline of the news article</li>
            <li>article_link: link to the original news article. Useful for collecting supplementary data</li>
        </ul>
        </div>
        """
        st.markdown(html3,unsafe_allow_html=True)
        

    st.sidebar.header('Developed by')
    html_string = "<div><a href='https://ritwiksharma107.github.io/portfolio/' style='color:#e60067; text-decoration:none;'>Ritwik Sharma</a></div>"
    st.sidebar.markdown(html_string, unsafe_allow_html=True)

    page_bg_img = '''
    <style>
    body {
    background-image: url("https://www.wallpapertip.com/wmimgs/172-1722866_wallpaper-gradient-pink-white-linear-light-pink-light.jpg");
    background-size: cover;
    }
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)

if __name__ == '__main__':
    main()