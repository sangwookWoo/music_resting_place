from googletrans import Translator
import pandas as pd
translator = Translator()

df = pd.read_csv('playdata_project/music_resting_place/Recommend_songs/lyrics_v2.csv', encoding='utf-8')
# print(df.iloc[170])
for i in range(0, 171):
    input_string = df.iloc[i, 1]
    output_string = translator.translate(input_string, dest='ko', src='en')
    df.iloc[i, 1] = output_string.text
df.to_csv('playdata_project/music_resting_place/Recommend_songs/translated_lyrics_v2.csv', index=False)