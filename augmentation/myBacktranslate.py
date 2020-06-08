import csv
import torch

input_file = 'aug_1.txt'
output_file = 'aug_2.txt'

en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')

with open(input_file) as f:
    with open(output_file, 'w') as o:
        for line in f:
            line = line.replace('\n', '')
            # Write the original sentence
            o.write(line + '\n')
            # write paraphrase one
            paraphrase = de2en.translate(en2de.translate(line))
            if line != paraphrase:
                o.write(paraphrase + '\n')