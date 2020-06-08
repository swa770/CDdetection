import os
import csv

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action

input_file = 'should_data.txt'
output_file = 'aug_1.txt'

with open(input_file) as f:
    with open(output_file, 'w') as o:
        for line in f:
            line = line.replace('\n', '')
            # Write the original sentence
            o.write(line + '\n')

            # Substitute word by word2vec similarity
            ## Synonym
            # WordNet
            aug = naw.SynonymAug(aug_src='wordnet')
            augmented_text = aug.augment(line)
            if line != augmented_text:
                o.write(augmented_text + '\n')

            ## Contextual Word Insertion/Replacement
            # Insert word by contextual word embeddings (BERT, DistilBERT, RoBERTA or XLNet)
            aug = naw.ContextualWordEmbsAug(
                model_path='bert-base-uncased', action="insert")
            augmented_text = aug.augment(line)
            if line != augmented_text:
                o.write(augmented_text + '\n')

            # Substitute word by contextual word embeddings (BERT, DistilBERT, RoBERTA or XLNet)
            aug = naw.ContextualWordEmbsAug(
                model_path='bert-base-uncased', action="substitute")
            augmented_text = aug.augment(line)
            if line != augmented_text:
                o.write(augmented_text + '\n')

# model_type: word2vec, glove or fasttext
'''
aug = naw.WordEmbsAug(
    model_type='word2vec', model_path='GoogleNews-vectors-negative300.bin',
    action="substitute")
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)
'''

# PPDB
'''
aug = naw.SynonymAug(aug_src='ppdb', model_path=os.environ.get("MODEL_DIR") + 'ppdb-2.0-tldr')
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)
'''