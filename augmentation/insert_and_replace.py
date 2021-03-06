import os
import csv

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action

# paths for Input and Output files
# must be separated by \n
input_file = 'data.txt'
output_file = 'aug_1.txt'

# generate 3 different versions of sentences
with open(input_file) as f:
    with open(output_file, 'w') as o:
        for line in f:
            line = line.replace('\n', '')
            # Write the original sentence
            o.write(line + '\n')

            ## Synonym
            # WordNet
            aug = naw.SynonymAug(aug_src='wordnet')
            augmented_text = aug.augment(line)
            if line != augmented_text:
                o.write(augmented_text + '\n')

            ## Contextual Word Insertion/Replacement
            # Insert word by contextual word embeddings BERT
            aug = naw.ContextualWordEmbsAug(
                model_path='bert-base-uncased', action="insert")
            augmented_text = aug.augment(line)
            if line != augmented_text:
                o.write(augmented_text + '\n')

            # Substitute word by contextual word embeddings BERT
            aug = naw.ContextualWordEmbsAug(
                model_path='bert-base-uncased', action="substitute")
            augmented_text = aug.augment(line)
            if line != augmented_text:
                o.write(augmented_text + '\n')
