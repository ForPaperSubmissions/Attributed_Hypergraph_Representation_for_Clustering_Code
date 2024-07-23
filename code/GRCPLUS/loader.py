from dataset import (
    NewsDataset,
    MushroomDataset,
    NTU2012Dataset,
    ModelNet40Dataset,
    CociWikiDataset,
    CoauCoraDataset,
    CoauDblpDataset,
    CociCoraDataset,
    CociCiteataset,
    CociPubmtaset,
    News20Dataset,
    CociPatentC13,
)


class DatasetLoader(object):
    def __init__(self):
        pass

    def load(self, dataset_name: str = 'cora'):
        if dataset_name == '20newsW100':
            return NewsDataset()
        elif dataset_name == 'Mushroom':
            return MushroomDataset()
        elif dataset_name == 'NTU2012':
            return NTU2012Dataset()
        elif dataset_name == 'ModelNet40':
            return ModelNet40Dataset()
        elif dataset_name == 'coci_wiki':
            return CociWikiDataset()
        elif dataset_name == 'coau_cora':
            return CoauCoraDataset()
        elif dataset_name == 'coau_dblp':
            return CoauDblpDataset()
        elif dataset_name == 'coci_cora':
            return CociCoraDataset()
        elif dataset_name == 'coci_citeseer':
            return CociCiteataset()
        elif dataset_name == 'coci_pubmed':
            return CociPubmtaset()
        elif dataset_name == '20news':
            return News20Dataset()
        elif dataset_name == 'coci_patent_C13':
            return CociPatentC13()
        else:
            assert False
