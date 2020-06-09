CS263 Final Project: Detech Cognitive Distortion

Members: Jeannie Huang, Rohan Dutta, Su Yong Won

References:
1. BERT: https://modelzoo.co/model/pytorch-pretrained-bert
2. nlpaug: https://github.com/makcedward/nlpaug
3. BERT Implementation: https://github.com/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb

Datasets:
1. Sentiment140 http://help.sentiment140.com/for-students
2. In the CS263_Project section we have:
  2.1. all_data ( augmented positive class and negative class labels)
  2.2. aug_data (augmented negative class labels)
  2.3. Labeled Data (Sheet with positive class, negative class data and results)
  2.4. test_set (Unbiased completely different dataset, which has not been augmented, 50 enties)
  2.5. sentiment (sentiment data)

Models:

Can be found in the CS263_Project/Best_Models section. We have three pickle files with the best model obtained after hyper parameter tuning:
1. best_clf_bert
2. best_clf_distilbert
3. best_clf_roberta

Also the classifier models for sentiment analysis can be found in CS263_Project/Sentiment_models:
1. sentiment_model_bert
2. sentiment_model_distil
3. sentiment_model_roberta


Codes:

Can be found in the CS263_Project/codes section. The following ipython notebooks can be run as is on google colaboratory/ jupyter notebook:
1. CS263_Project_BERT.ipynb
2. CS263_Project_DistilBERT.ipynb
3. CS263_RoBERTA.ipynb
