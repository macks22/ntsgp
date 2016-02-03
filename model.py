"""
Model and Results classes for encapsualting ML method train and predict logic
as well as prediction evaluation logic.
"""
import inspect

import naming

class Model(object):
    """Encapsulate model with known API for TrainTestSplit use."""

    def __init__(self, model, splitter):
        self.model = model
        self.splitter = splitter

    @property
    def model_name(self):
        return self.model.__class__.__name__

    @property
    def model_suffix(self):
        """All scikit-learn estimators have a `get_params` method."""
        return naming.suffix_from_params(self.model.get_params())

    @staticmethod
    def func_kwargs(func):
        argspec = inspect.getargspec(func)
        if argspec.defaults is None:
            return argspec.args
        else:
            return argspec.args[-len(argspec.defaults):]

    @staticmethod
    def func_pargs(func):
        argspec = inspect.getargspec(func)
        arglist = argspec.args[1:]  # remove self
        if argspec.defaults is None:
            return arglist
        else:
            return arglist[:-len(argspec.defaults)]


class SklearnModel(Model):
    """Encapsulate estimator with scikit-learn API for TrainTestSplit use."""

    @property
    def fit_kwargs(self):
        """Keyword arguments to model `fit` method."""
        return self.func_kwargs(self.model.fit)

    @property
    def fit_pargs(self):
        """Positional arguments to model `fit` method."""
        return self.func_pargs(self.model.fit)

    @property
    def predict_kwargs(self):
        """Keyword arguments to model `predict` method."""
        return self.func_kwargs(self.model.predict)

    @property
    def predict_pargs(self):
        """Positional arguments to model `predict` method."""
        return self.func_pargs(self.model.predict)


class SklearnRegressionModel(SklearnModel):

    def fit_predict(self, val):
        split = self.splitter[val]
        train_X, train_y, train_eids,\
        test_X, test_y, test_eids, indices, nents = \
            split.preprocess(all_null='drop')

        # Create copy of model with same params.
        model = self.model.__class__(**self.model.get_params())

        # Pass in only keyword arguments the model actually needs for fitting.
        # This is accomplished via function argspec inspection.
        all_kwargs = {'entity_ids': train_eids,
                      'feature_indices': indices,
                      'n_entities': nents}
        kwargs = {k: v for k, v in all_kwargs.items() if k in self.fit_kwargs}
        model.fit(train_X, train_y, **kwargs)

        # Make predictions using the learned model.
        kwargs = {k: v for k, v in all_kwargs.items()
                  if k in self.predict_kwargs}
        pred_y = model.predict(test_X, **kwargs)
        return RegressionResults(pred_y, split.test, split.fguide, model)

    def fit_predict_all(self, n_jobs=1):
        """Run sequential fit/predict loop for all possible data splits in a
        generative manner.

        This should eventually account for:

        1.  parallel evaluation if using batch methods
        2.  results not fitting in memory
        3.  optional reassembly of full data frame with original and predicted
            results.
        """
        pass


class Results(object):
    """Encapsulate model prediction results & metadata for evaluation."""

    def __init__(self, predicted, test_data, fguide, model):
        self.fguide = fguide
        self.pred_colname = '%s_predicted' % fguide.target
        self.test_data = test_data
        self.test_data.loc[:, self.pred_colname] = predicted
        self.model = model

    @property
    def predicted(self):
        return self.test_data[self.pred_colname]

    @property
    def actual(self):
        return self.test_data[self.fguide.target]

    @property
    def model_params(self):
        """Return model params learned during fitting."""
        params = [attr for attr in dir(self.model)
                  if not attr.endswith('__') and attr.endswith('_')]
        return {param: getattr(self.model, param) for param in params}


class RegressionResults(Results):
    """Encapsulate model regression predictions & metadata for evaluation."""

    def error(self):
        return self.predicted - self.actual

    def squared_error(self):
        return self.error() ** 2

    def sse(self):
        return self.squared_error().sum()

    def mse(self):
        return self.squared_error().mean()

    def rmse(self):
        return np.sqrt(self.mse())

    def mae(self):
        return abs(self.error()).mean()

