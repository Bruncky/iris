import joblib

from iris.data import get_data, holdout
from iris.pipeline import TaxiFarePipeline

class Trainer():
    def __init__(self):
        pass

    def fit(self):
        self.pipeline.fit(self.X_train, self.y_train)

    def save_pipeline(self):
        joblib.dump(self.pipeline, 'pipeline.joblib')

    def train(self):
        df = get_data()

        self.X_train, self.X_test, self.y_train, self.y_test = holdout(df)

        iris_pipeline = TaxiFarePipeline()
        self.pipeline = iris_pipeline.create_pipeline()

        self.fit()

        self.save_pipeline()

if __name__ == '__main__':
    Trainer().train()