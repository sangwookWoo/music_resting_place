import pandas as pd
import streamlit as st
from streamlit_chat import message
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn.functional as F
import json
from catboost import CatBoostClassifier
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import os
from model_predict import *

# 경로 지정
filePath, fileName = os.path.split(__file__)

# 토크나이저 로드
@st.cache(allow_output_mutation = True)
def tokenizer_load():
    with open(os.path.join(filePath, 'models', 'tokenizer.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)
        return tokenizer
    
# 모델 로드
@st.cache(allow_output_mutation = True)
def predict_model_load():
    predict_model = CatBoostClassifier()
    predict_model.load_model(os.path.join(filePath, 'models', 'catboost.cbm'))
    return predict_model

@st.cache(allow_output_mutation = True)
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache(allow_output_mutation = True)
def get_dataset():
    df = pd.read_parquet(os.path.join(filePath, 'data', 'WellnessData.parquet'), engine='pyarrow') 
    # df['embedding'] = df['embedding'].apply(json.loads)
    return df

def main():
    
    # 페이지 세팅
    st.set_page_config(page_title = "음악 쉼표", layout='wide', initial_sidebar_state='collapsed')
    
    # 모델 불러오기
    model = cached_model()
    
    # 토크나이저, 예측 모델 불러오기
    tokenizer = tokenizer_load()
    predict_model = predict_model_load()
    
    
    # 데이터셋 불러오기
    df = get_dataset()

    # st.markdown('## 지금 느끼는 감정들을 이야기해주세요 😊')
    st.markdown("<h2 style='text-align: center; color: black;'>지나치는 감정들과 일상들을 이야기해주세요 😊</h2>", unsafe_allow_html=True)
    st.write(' ')

    # 두 구역으로 나눔
    visualization, chatbot = st.columns(2)

    with chatbot:
        # 세션스테이트에 세션이 재실행되도 초기화 되지 않게끔 세션 설정
        if 'generated' not in st.session_state:
            st.session_state['generated'] = []
            
        if 'past' not in st.session_state:
            st.session_state['past'] = []

                
        # 채팅 폼 만들기
        with st.form('form', clear_on_submit = True): # 제출하면 텍스트 박스 지워지게 만들기
            user_input = st.text_input('고민하는 나 : ', '')
            submitted = st.form_submit_button('전송')
            
        if submitted and user_input: # 제출과 user_input 값이 True면 임베딩을 통해 최고 유사도 답변 추출
            # 모델 임베딩
            embedding = model.encode(user_input)
            
            # 임베딩 한 것 중 코사인 유사도 계산
            # df['simillarity'] = df['embedding'].apply(lambda x : cosine_similarity([embedding], [x]).squeeze())
            df['simillarity'] = F.cosine_similarity(torch.FloatTensor(embedding * len(df['embedding'])),  torch.FloatTensor(df['embedding']))

            # 가장 유사한 답변 추출
            answer = df.loc[df['simillarity'].idxmax()]
            
            st.session_state['past'].append(user_input)
            
            # 유사도 상 0.72 미만이면 질문 하는 응답지로 넘어감.
            if answer['simillarity'] < 0.64:
                st.session_state['generated'].append('당신의 기분을 조금 더 알고 싶어요.')
            else:
                st.session_state['generated'].append(answer['A'])


        for i in range(len(st.session_state['past']) - 1, -1, -1):
            message(st.session_state['past'][i], is_user = True, key = str(i) + '_user', )
            message(st.session_state['generated'][i], key = str(i) + '_bot')
            
    user_text = ' '.join(st.session_state['past'])
    st.write(user_text)
    with visualization:
        user_text = ' '.join(st.session_state['past'])
        user_length = len(user_text)
        # st.write(user_length / 300)
        
        # 노래 추천 누르면 catboost 모델 작동, 아웃풋은 predict_proba
        if st.button('노래 추천받기'):
            emotion, emotion_proba = predict_value(user_text, predict_model, tokenizer)
            # st.write(emotion_proba)
            st.video('https://www.youtube.com/watch?v=R8axRrFIsFI')
        
    # # 텍스트 저장
    # st.write(st.session_state['past'])
    user_text = ' '.join(st.session_state['past'])

if __name__ == "__main__":
    main()