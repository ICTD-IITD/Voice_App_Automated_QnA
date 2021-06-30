# Early Results from Automating Voice-based Question-Answering Services Among Low-income Populations in India

This is the repository containing the code for our work here (add link).

Instructions for running:

Our code can be used for identifying the query in an FAQ database, which is most similar to the input test query. It can concatenate multiple datasets, conduct data augmentation by generation of similar sentences and also perform automatic theme classification. We hope our contribution helps future researchers in providing a headstart on experiments. Please add your data inside the data directory and modify the code for reading them appropriately.

1. bring_in_stt.py : generates the final csv file used for pre-processing data. This will take in the speech to text transcripts (STTs) from the respective excel file and combine them with the dataset containing all other information.
2. preprocess_data.py : generates train and test splits. Supports multiple train sets for a single test set (make appropriate changes in config.py, instructions are self explanatory)
3. All the notebooks correspond to different models/libraries used and describe their workflow in detail. They can be run directly by using Google Colab using the link present in the notebooks for ease. The notebooks download this repository, and then use the data present in the data folder for training, testing and evaluation. 
4. For data augmentation and data concatenation, refer to data_concatenation_augmentation.ipynb : this can both concatenate and augment datasets, by creating similar sentences using manually crafted synonyms and using iNLTK library's api call for similar sentence generation.
5. Theme Classification: theme_classification.ipynb contains code to experiment with BERT and Tf-Idf weighted N-gram models which will predict the theme of the input test query (q2), and then generate a filtered test data where q1's theme will be among the top 3 predicted themes for q2.
 
