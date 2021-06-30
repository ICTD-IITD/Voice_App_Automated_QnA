import pandas as pd
from bs4 import BeautifulSoup
import requests

def get_unique_index(url):
    if url.endswith('.mp3'):
        return url
    r = requests.get(url)
    
    soup = BeautifulSoup(r.content,'html.parser')
    links = soup.findAll('a')
    mp3_links = [link['href'] for link in links if link['href'].endswith('mp3')]
    if len(mp3_links) == 0:
        print(url)
        return url
    link=mp3_links[0]
    return 'http://voice.gramvaani.org' + link

df_stt = pd.read_excel('data/Q_A_STT_Transcribed.xlsx', sheet_name=0)
df = pd.read_excel('data/jeevika_final.xlsx', sheet_name=0).fillna('')
df_stt2 = pd.read_excel('data/jeevika_stt.xlsx', sheet_name=0)
df['STT Transcript'] = df['Caller query']
need_stt = []
for i,r in df.iterrows():
    df_ = df_stt[df_stt['Caller query'] == r['Caller query']]
    df_2 = df_stt2[(df_stt2['Public Link'] == r['Caller query'])|(df_stt2['Recording audio link'] == r['Caller query'])]
    if len(df_) > 0:
        stt = list(df_['STT Transcript'])[0]
        df.loc[i, 'STT Transcript'] = stt
    elif len(df_2) > 0:
        stt = list(df_2['ML transcript'])[0]
        df.loc[i, 'STT Transcript'] = stt
    elif df.loc[i, 'Caller query']!='':
        df.loc[i, 'STT Transcript'] = df.loc[i, 'Caller query transcription']
        need_stt.append(get_unique_index(df.loc[i, 'Caller query']))
        
df.to_excel('data/jeevika_final_stt.xlsx', index = False)      
df_ = pd.DataFrame({'Caller query': need_stt})
df_.to_excel('data/need_jee_stt.xlsx', index = False)

df_stt = pd.read_excel('data/paj_stt.xlsx', sheet_name=0)

df = pd.read_excel('data/paj_final.xlsx', sheet_name=0).fillna('')

df['STT Transcript'] = df['Caller query']
need_stt = []
for i,r in df.iterrows():
    df_ = df_stt[(df_stt['Public Link'] == r['Caller query'])|(df_stt['Recording audio link'] == r['Caller query'])]
    if len(df_) > 0:
        stt = list(df_['ML transcript'])[0]
        df.loc[i, 'STT Transcript'] = stt
    elif df.loc[i, 'Caller query']!='':
        df.loc[i, 'STT Transcript'] = df.loc[i, 'Caller query transcription']
        need_stt.append(get_unique_index(df.loc[i, 'Caller query']))
        
df.to_excel('data/paj_final_stt.xlsx', index = False)      
df_ = pd.DataFrame({'Caller query': need_stt})
df_.to_excel('data/need_stt_paj.xlsx', index = False)
