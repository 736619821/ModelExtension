from fairseq.models.roberta import RobertaModel
from scipy.stats import pearsonr

# roberta = RobertaModel.from_pretrained(
#     '/mnt/vol1/gaowenchun/new/roberta_test/roberta-model/checkpoints/',
#     checkpoint_file='checkpoint_test.pt',
#     data_name_or_path='STS-B-bin'
# )

# roberta = RobertaModel.from_pretrained(
#     '/mnt/vol1/gaowenchun/new/roberta_test/roberta-model/roberta.large/',
#     checkpoint_file='model.pt',
#     data_name_or_path='STS-B-bin'
# )

# roberta = RobertaModel.from_pretrained(
#     '/mnt/vol1/gaowenchun/new/roberta_test/fairseq-0.9.0_1/checkpoints/',
#     checkpoint_file='checkpoint10.pt',
#     data_name_or_path='STS-B-bin'
# )

roberta = RobertaModel.from_pretrained(
    '/mnt/vol1/gaowenchun/new/roberta_test/ELLE-main/fairseq-0.9.1/checkpoints/',
    checkpoint_file='checkpoint8.pt',
    data_name_or_path='STS-B-bin'
)


roberta.cuda()
roberta.eval()
gold, pred = [], []
ncorrect, nsamples = 0, 0
with open('/mnt/vol1/gaowenchun/new/roberta_test/roberta-data/newstsb/STS-B/dev.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        #print(tokens)
        sent1, sent2, target = tokens[7], tokens[8], float(tokens[9])
        tokens = roberta.encode(sent1, sent2)
        features = roberta.extract_features(tokens)
        #print(roberta.model)
        predictions = 5*roberta.model.classification_heads['sentence_classification_head'](features)
        gold.append(target)
        pred.append(predictions.item())
        print(target, predictions.item(), round(target), round(predictions.item()))
        if abs(predictions.item() - target) <= 1:
            ncorrect += 1
        #ncorrect += int(round(predictions.item()) == round(target))
        nsamples += 1

print('| Pearson: ', pearsonr(gold, pred))
print('| Accuracy: ', float(ncorrect)/float(nsamples))
