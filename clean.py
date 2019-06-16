import pandas as pandas
import pdb


def convert_to_bert(input)
    raw_df = pandas.read_csv(input, header=None)
    return pandas.DataFrame({
        'id': range(len(raw_df)),
        'label': raw_df[0],
        'alpha': ['a']*raw_df.shape[0],
        'text': raw_df[1].replace(r'\n', ' ', regex=True)
    })

train_df = convert_to_bert('data/train.csv')
dev_df = convert_to_bert('data/test.csv')
