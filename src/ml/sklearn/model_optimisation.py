"""
Module for classification model optimization
"""
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import svm


class ImageModelOptimiser:
    """
    Image classifier model optimization
    """

    def __init__(self, model):
        self.HOG_pipeline = Pipeline([
            ('grayify', model.grayify),
            ('hogify', model.hogify),
            ('scalify', model.scalify),
            ('classify', model.classifier)
        ])

        self.param_grid = [
            {
                'hogify__orientations': [8, 9],
                'hogify__cells_per_block': [(2, 2), (3, 3)],
                'hogify__pixels_per_cell': [(8, 8), (10, 10), (12, 12)]
            },
            {
                'hogify__orientations': [8],
                'hogify__cells_per_block': [(3, 3)],
                'hogify__pixels_per_cell': [(8, 8)],
                'classify': [
                    model.classifier,
                    svm.SVC(kernel='linear')
                ]
            }
        ]

    def optimize(self, X, y):
        grid_search = GridSearchCV(self.HOG_pipeline,
                                   self.param_grid,
                                   cv=3,  # cross validation: 3 folds
                                   n_jobs=-1,
                                   scoring='accuracy',
                                   verbose=1,
                                   return_train_score=True)

        return grid_search.fit(X, y)
