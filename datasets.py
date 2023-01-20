from data_utils import (
    get_input_mask, pseudo_summary_f1, shift_tokens_right,
    padding_to_maxlength, load_stopwords, text_segmentate)

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, file,self_generated):
        self.ann_path = file
        self.self_generaed = self_generated
        self.ann = open(self.ann_path, 'r',encoding='utf-8')
        self.data=self.ann.readlines()
        if self.self_generaed != None:
            self.self_generateddata_file=open(self.self_generaed,'r',encoding='utf-8')
            self.self_generateddata=self.self_generateddata_file.readlines()
            idx=[]
            example=[]
            for i in range(len(self.data)):
                idx.append(i)
                example.append(self.data[i])
            for i in range(len(self.self_generateddata)):
                idx.append(i)
                example.append(self.self_generateddata[i])
        else:
            idx=[]
            example=[]
            for i in range(len(self.data)):
                idx.append(i)
                example.append(self.data[i])
        self.examples=dict(zip(idx,example))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        sample = self.examples[idx]

        return sample