import glob
import json
import os
from torch.utils.data import Dataset


class Wikipedia(Dataset):
    def __init__(self, root_dir):
        assert os.path.exists(root_dir)
        self.data_dir = root_dir
        self.docs = self._read_all()

    def _read_all(self):
        dir_list = self._list_dir(self.data_dir)
        doc_list = []
        for dir_i in dir_list:
            sublist = self._read_dir(dir_i)
            for i in sublist:
                doc_list.append(i)
        return doc_list

    def _read_dir(self, dir_i):
        files = self._list_dir(dir_i)
        list = []
        for file in files:
            sublist = self._read_file(file)
            for i in sublist:
                list.append(i)
        return list

    def _list_dir(self, dir_i):
        files = glob.glob(os.path.abspath(dir_i) + '/*')
        files.sort()
        return files

    def _read_file(self, file):
        d = []
        with open(file, 'r') as f:
            for line in f.readlines():
                d_i = json.loads(line)
                d_i['text'] = list(filter(None, d_i['text'].split('\n')))
                d.append(d_i)
        return d
