from libcity.data.dataset import ClusterDataset, DownStreamSubDataset
from torch.utils.data import DataLoader


class SimilarityDataset(ClusterDataset):
    def __init__(self, config):
        super().__init__(config)
        self.origin_data_path = config.get('query_data_path', None)
        self.detour_data_path = config.get('detour_data_path', None)
        self.origin_big_data_path = config.get('origin_big_data_path', None)

    def _gen_dataset(self):
        database_dataset = DownStreamSubDataset(data_name=self.dataset,
                                                data_path=self.origin_big_data_path,
                                                vocab=self.vocab, seq_len=self.seq_len, add_cls=self.add_cls,
                                                max_train_size=None,
                                                geo2latlon=self.geoid2latlon)
        detour_dataset = DownStreamSubDataset(data_name=self.dataset,
                                              data_path=self.detour_data_path,
                                              vocab=self.vocab, seq_len=self.seq_len, add_cls=self.add_cls,
                                              max_train_size=None,
                                              geo2latlon=self.geoid2latlon)
        query_dataset = DownStreamSubDataset(data_name=self.dataset,
                                             data_path=self.origin_data_path,
                                             vocab=self.vocab, seq_len=self.seq_len, add_cls=self.add_cls,
                                             max_train_size=None,
                                             geo2latlon=self.geoid2latlon)
        return database_dataset, detour_dataset, query_dataset

    def _gen_dataloader(self, database_dataset, detour_dataset, query_dataset):
        database_dataloader = DataLoader(database_dataset, batch_size=self.batch_size,
                                         num_workers=self.num_workers, shuffle=False,
                                         collate_fn=lambda x: self.collate_fn(
                                            x, max_len=self.seq_len, vocab=self.vocab, add_cls=self.add_cls))
        detour_dataloader = DataLoader(detour_dataset, batch_size=self.batch_size,
                                       num_workers=self.num_workers, shuffle=False,
                                       collate_fn=lambda x: self.collate_fn(
                                           x, max_len=self.seq_len, vocab=self.vocab, add_cls=self.add_cls))
        query_dataloader = DataLoader(query_dataset, batch_size=self.batch_size,
                                      num_workers=self.num_workers, shuffle=False,
                                      collate_fn=lambda x: self.collate_fn(
                                           x, max_len=self.seq_len, vocab=self.vocab, add_cls=self.add_cls))
        return database_dataloader, detour_dataloader, query_dataloader
