
#===================================================================#
#                   General Configuration                           #
#===================================================================#
"""

For training on multiple datasets, add excel files to the list. But the test questions and all_data.csv files are generated only 
from the first excel sheet in the list. Make sure all the questions are present in the first sheet of the excel file, and column names
are present in the format given below in COLUMN_NAMES

KAB : 'data/KAB QA data with all manual transcripts (with similar).xlsx' 
Jeevika : 'data/Q&A Transcribed (compiled) (2).xlsx'
Old Jeevika with STT: 'data/Q_A_STT_Transcribed.xlsx',
New Jeevika with STT: 'data/jeevika_final_stt.xlsx'
"""
# For train and test collumn, choose one among 'Caller query transcription'/ 'Relevant Topic' / 'STT Transcript'
EXCEL_FILE_PATH = ['data/jeevika_final_stt.xlsx']
DATA = 'jee'
TRAIN_COLUMN = 'Caller query transcription'
TEST_COLUMN = 'STT Transcript' 
MAX_LEN = 256
LEARNING_RATE_1 = 2e-6
LEARNING_RATE_2 = 2e-5
EPOCH_NUM = 6
BATCH_SIZE = 8
TEST_BATCH_SIZE = 50
MODEL_NAME = 'BERT'

#===================================================================#
#                   Excel Sheet Configuration                       #
#===================================================================#

COLUMN_NAMES = {
    'BROAD_THEME' : 'Broad theme',
    'DETAIL' : 'Detail',
    
    'RELEVANT_TOPIC' : 'Relevant Topic',
#     'RELEVANT_POINT' : 'Answer Transcription',
    'RELEVANT_POINT' : 'Relevant Point',
    'STT' : 'STT Transcript',
    'QUESTION_URL' : 'Caller query',
    'ANSWER_URL' : 'Answers',
    
    'A_TRANSCRIPTION' : 'Answer Transcription',
    'Q_TRANSCRIPTION' : 'Caller query transcription'

}
