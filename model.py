"""
Model and Results classes for encapsualting ML method train and predict logic
as well as prediction evaluation logic.
"""
import os
import inspect
import warnings
import importlib

import naming
import saveload


class Model(object):
    """Encapsulate model with known API for TrainTestSplit use."""

    def __init__(self, model):
        self.model = model

    @property
    def model_name(self):
        return self.model.__class__.__name__

    @property
    def model_suffix(self):
        """All scikit-learn estimators have a `get_params` method."""
        return naming.suffix_from_params(self.fixed_params())

    @property
    def fixed_params(self):
        raise NotImplementedError('fixed_params not implemented')

    @property
    def learned_params(self):
        raise NotImplementedError('learned_params not implemented')

    @property
    def all_params(self):
        params = self.fixed_params
        params.update(self.learned_params)
        return params

    def clone(self):
        return self.__class__(**self.fixed_params)

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

    def save(self, savedir, ow=False):
        params = {
            'fixed': self.fixed_params,
            'learned': self.learned_params,
            'metadata': {
                'name': self.model_name,
                'module': self.model.__module__
            }
        }
        saveload.save_var_tree(params, savedir, ow)

    @classmethod
    def load(cls, savedir):
        params = saveload.load_var_tree(savedir)
        model_module = importlib.import_module(params['metadata']['module'])
        model_class = getattr(model_module, params['metadata']['name'])
        inner_model = model_class(**params['fixed'])

        model = cls(inner_model)
        for learned_param, val in params['learned'].items():
            setattr(model, learned_param, val)
        return model


class SklearnModel(Model):
    """Encapsulate estimator with scikit-learn API for TrainTestSplit use."""

    @property
    def fixed_params(self):
        return self.model.get_params()

    @property
    def learned_params(self):
        params = [attr for attr in dir(self.model)
                  if not attr.endswith('__') and attr.endswith('_')]

        # In older scikit-learn versions, some of the learned parameters (those
        # ending in _) were set in the __init__ method. As of v0.17, these are
        # deprecated, and they will be removed in v0.19. We suppress the
        # warnings here and catch the AttributeErrors that pop up when these
        # parameters are implemented as properties and not initially set.
        available_params = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for param in params:
                try:
                    available_params[param] = getattr(self.model, param)
                except AttributeError:
                    pass

        return available_params

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

    def fit(self, X, y, *args, **kwargs):
        pass

    def predict(self, X, y, *args, **kwargs):
        pass


class SklearnRegressionRunner(object):

    def __init__(self, model, splitter):
        """Wrap up a Model with a TrainTestSplitter with methods for training
        the model on the various train/test splits produced by the splitter.

        Args:
            model (Model): The model to train and predict with.
            splitter (TrainTestSplitter): The splitter to use for producing
                train/test data splits.
        """
        self.model = model
        self.splitter = splitter

    def fit_predict(self, val):
        """Get the train/test set for `val`, train a copy of the model with the
        same parameters on the train set, and then predict for the test set.
        Return a RegressionResults object with the results, including the
        learned model parameters.
        """
        split = self.splitter[val]
        train_X, train_y, train_eids,\
        test_X, test_y, test_eids, indices, nents = \
            split.preprocess(all_null='drop')

        # Create copy of model with same params.
        model = self.model.clone()

        # Pass in only keyword arguments the model actually needs for fitting.
        # This is accomplished via function argspec inspection.
        all_kwargs = {'entity_ids': train_eids,
                      'feature_indices': indices,
                      'n_entities': nents}
        kwargs = {k: v for k, v in all_kwargs.items()
                  if k in self.model.fit_kwargs}
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

