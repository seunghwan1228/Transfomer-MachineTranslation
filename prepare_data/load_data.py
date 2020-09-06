import tensorflow_datasets as tfds



# The Dataset from tfds
class LoadData:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name


    def get_data(self):
        examples, info = tfds.load(str(self.dataset_name),
                                with_info=True,
                                as_supervised=True)
        print(examples)
        train_data = examples['train']
        valid_data = examples['validation']
        test_data = examples['test']

        return train_data, valid_data, test_data, info




if __name__ == '__main__':
    dataset_name = 'ted_hrlr_translate/pt_to_en'
    data_loader = LoadData(dataset_name)
    a, b, c, d= data_loader.get_data()


    for n, (lang1, lang2) in enumerate(a):
        print(n)
        print(lang1)
        print(lang2)
        print()