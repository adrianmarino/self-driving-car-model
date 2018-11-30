

class DataGeneratorFactory:

    @staticmethod
    def train_generator(
            work_path,
            features,
            labels,
            batch_size
    ): return DataGeneratorFactory.batch_generator(work_path, features, labels, batch_size, True)

    @staticmethod
    def validation_generator(
            work_path,
            features,
            labels,
            batch_size
    ): return DataGeneratorFactory.batch_generator(work_path, features, labels, batch_size, False)


    @staticmethod
    def batch_generator(
            work_path,
            features,
            labels,
            batch_size,
            is_training
    ):
        return None