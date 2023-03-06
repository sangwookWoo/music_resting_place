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
import random
import psycopg2
import time
from pyecharts import options as opts
from pyecharts.charts import Bar
from streamlit_echarts import st_echarts


# 경로 지정
filePath, fileName = os.path.split(__file__)

# 데이터베이스 연결
def init_connection():
    return psycopg2.connect(**st.secrets["postgres"])

conn = init_connection()

# 쿼리 실행(지금은 insert, update, delte만 가능하게)
def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        conn.commit()

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
    add_question_df = pd.read_csv(os.path.join(filePath, 'data', 'chatbot_emotion_Q_list.csv'), encoding = 'cp949')
    # df['embedding'] = df['embedding'].apply(json.loads)
    return df, add_question_df

def main():
    
    # 페이지 세팅
    st.set_page_config(page_title = "음악 쉼표", layout='wide', initial_sidebar_state='collapsed')
    
    # 모델 불러오기
    model = cached_model()
    
    # 토크나이저, 예측 모델 불러오기
    tokenizer = tokenizer_load()
    predict_model = predict_model_load()
    
    
    # 데이터셋 불러오기
    df, add_question_df = get_dataset()

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
            
            # 유사도 상 0.64 미만이면 질문 하는 응답지로 넘어감. 0.64
            if answer['simillarity'] < 0.64:
                
                text_list = ('제가 당신에게 힘이 되는 비밀 친구가 되어 드릴게요.',
                            '꺼내고 싶은 마음을 얘기해주면 제가 열심히 들을게요',
                            '지금 느끼시는 감정을 조금 더 알려주세요',
                            '제가 당신에게 힘이 되는 비밀 친구가 되어 드릴게요.',
                            '꺼내고 싶은 마음을 얘기해주면 제가 열심히 들을게요',
                            '저는 항상 여기 있어요. 하고 싶은 이야기가 있다면 들려주시겠어요?',
                            '저는 들을 준비가 되어 있어요.')
                
                text_length = len(text_list) - 1
                text = text_list[random.randint(0, text_length)]
                
                # 특정 상황에서 답변 변경(도희)
                
                for idx, i in enumerate(add_question_df['chatbot_answer']):
                    if i in answer['A']:
                        text = add_question_df.at[idx,'add_question']
                        break
                st.session_state['generated'].append(text)
                
                # st.session_state['generated'].append('지금 느끼시는 감정을 조금 더 알려주세요')
                
            else:
                st.session_state['generated'].append(answer['A'])


        for i in range(len(st.session_state['past']) - 1, -1, -1):
            message(st.session_state['past'][i], is_user = True, key = str(i) + '_user', )
            message(st.session_state['generated'][i], key = str(i) + '_bot')
            
    user_text = ' '.join(st.session_state['past'])
    emotion, emotion_proba = predict_value(user_text, predict_model, tokenizer)
    
    with visualization:
        
        # 감정게이지
        user_length = len(user_text)
        
        tab1, tab2, tab3 = st.tabs(["💖감정 측정", "🔍내 감정과 유사한 곡을 듣고 싶어요", "🍀내 감정과 반대되는 곡을 듣고 싶어요"])
        # st.write(user_length / 300)
        
        sql_list = [0] # user_id를 넣을 예정(추후)
                
        # 최대 글자수
        base_len = 100
        
        total_length = int(user_length / base_len * 100)
        if total_length <= 100:
            tab1.markdown('##### 🎈 감정게이지')
            progress = tab1.progress(0)
            tab1.markdown('##### 당신의 감정이 차오르고 있어요!')
            progress.progress(total_length)
        else :
            tab1.markdown('##### 🎉 충분한 감정이 찼어요! 상태를 확인해보세요')
            if tab1.button('감정 상태 확인하기'):
                options = pie_chart(emotion_proba)
                st_echarts(options=options, height="500px")        
                            
        # 노래 추천 누르면 catboost 모델 작동, 아웃풋은 predict_proba
        # 글자 수가 100보다 클 때만 추천 버튼 활성화
        if total_length >= 0:
            if tab2.button('추천😊'):            
                # proba 값, 음악 넣기
                for x in emotion_proba.tolist()[0]:
                    sql_list.append(x)
                
                # 예측(코사인 유사도 기반)
                predict_cosine = cos_recommend(list(emotion_proba))
                
                # sql_list에 예측값 넣기
                sql_list.extend([predict_cosine[0], ''])
                
                # 노래 출력
                tab2.write(predict_cosine[0])
                tab2.video('https://www.youtube.com/watch?v=R8axRrFIsFI')
                
                # 쿼리 실행
                sql_query = f"insert into song.user_info (name, emotion0, emotion1, emotion2, emotion3, emotion4, song_sim, song_dif) values ({str(sql_list)[1:-1]});"
                run_query(sql_query)


        # 글자 수가 100보다 클 때만 추천 버튼 활성화
        if total_length >= 0:
            if tab3.button('추천😆'):
                # proba 값, 음악 넣기
                for x in emotion_proba.tolist()[0]:
                    sql_list.append(x)
                
                # 예측(코사인 유사도 기반)
                predict_cosine = cos_recommend(list(emotion_proba))
                
                # sql_list에 예측값 넣기
                sql_list.extend(['', predict_cosine[1]])

                # 노래 출력
                tab3.write(predict_cosine[1])
                tab3.video('https://www.youtube.com/watch?v=R8axRrFIsFI')
                
                # 쿼리실행
                sql_query = f"insert into song.user_info (name, emotion0, emotion1, emotion2, emotion3, emotion4, song_sim, song_dif) values ({str(sql_list)[1:-1]});"
                run_query(sql_query)
            
    # # 텍스트 저장
    # st.write(st.session_state['past'])

if __name__ == "__main__":
    main()