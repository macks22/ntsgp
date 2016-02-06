"""
Model and Results classes for encapsualting ML method train and predict logic
as well as prediction evaluation logic.
"""
import os
import inspect
import logging
import warnings
import importlib
import collections

import numpy as np

import mldata
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
        model_module = importlib.import_module(self.model.__module__)
        model_class = getattr(model_module, self.model_name)
        inner_model = model_class(**self.fixed_params)

        model = self.__class__(inner_model)
        for learned_param, val in self.learned_params.items():
            setattr(model, learned_param, val)
        return model

    def __eq__(self, other):
        my_params = self.all_params
        other_params = other.all_params
        if len(my_params) != len(other_params):
            return False

        for key in my_params.keys():
            my_val = my_params[key]
            if hasattr(my_val, 'shape'):  # comparing numpy arrays
                if not np.allclose(my_val, other_params[key]):
                    return False
            else:  # normal value comparison
                if not my_val == other_params[key]:
                    return False

        return True

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
                'class': self.__class__.__name__,
                'name': self.model_name,
                'module': self.model.__module__
            }
        }
        saveload.save_var_tree(params, savedir, ow)

    @classmethod
    def load(cls, savedir):
        """Load the model from the savedir.

        This assumes the Model subclass exists in this module.
        """
        params = saveload.load_var_tree(savedir)
        model_module = importlib.import_module(params['metadata']['module'])
        model_class = getattr(model_module, params['metadata']['name'])
        inner_model = model_class(**params['fixed'])

        outer_class = globals()[params['metadata']['class']]
        model = outer_class(inner_model)
        for learned_param, val in params['learned'].items():
            setattr(model, learned_param, val)
        return model


def key_intersect(dict1, dict2):
    """Return a dictionary with only the elements of the first dictionary that
    have keys which overlap with those of the second.
    """
    return {k: v for k, v in dict1.items() if k in dict2}


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
    def fitted(self):
        """Has the model been fitted?"""
        return len(self.learned_params) > 0

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

    def fit(self, X, y, **kwargs):
        """Fit the model. This method uses introspection to determine which
        keyword arguments are accepted and only passes those through.
        """
        filtered_kwargs = key_intersect(kwargs, self.fit_kwargs)
        self.model.fit(X, y, **filtered_kwargs)
        return self

    def predict(self, X, **kwargs):
        """Predict new target variables using the fitted model. This method
        uses introspection to determine which keyword arguments are accepted
        and only passes those through.
        """
        filtered_kwargs = key_intersect(kwargs, self.predict_kwargs)
        return self.model.predict(X, **filtered_kwargs)


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

    def fit_predict(self, split):
        """Take a TrainTestSplit and train a copy of the model with the same
        parameters on the train set, and then predict for the test set.  Return
        a RegressionResults object with the results, including the learned
        model parameters.

        Args:
            split (TrainTestSplit): Train on the training data and predict for
                the test data.
        Return: instance of ResultsSet.
        """
        train_X, train_y, train_eids,\
        test_X, test_y, test_eids, indices, nents = \
            split.preprocess(all_null='drop')

        # Create copy of model with same params.
        model = self.model.clone()

        # Extraneous kwargs are filtered by the fit/predict methods.
        kwargs = {'entity_ids': train_eids,
                  'feature_indices': indices,
                  'n_entities': nents}
        model.fit(train_X, train_y, **kwargs)
        pred_y = model.predict(test_X, **kwargs)
        return RegressionResults(pred_y, split.test, split.fguide, model)

    def fit_predict_for_value(self, val):
        """Get the train/test set for `val`, train a copy of the model with the
        same parameters on the train set, and then predict for the test set.
        Return a RegressionResults object with the results, including the
        learned model parameters.

        Args:
            val (object): The key to use to lookup the TrainTestSplit using the
                Splitter. For a PandasDataset, this is a value of the column
                name being used for splitting.
        Return: instance of ResultsSet.
        """
        split = self.splitter[val]
        return self.fit_predict(split, **kwargs)

    def fit_predict_all(self, n_jobs=1, errors='log'):
        """Run sequential fit/predict loop for all possible data splits in a
        generative manner.

        This should eventually account for:

        1.  parallel evaluation if using batch methods
        2.  results not fitting in memory
        3.  optional reassembly of full data frame with original and predicted
            results.

        Args:
            n_jobs (int): Number of jobs to use for parallel processing of the
                multiple TrainTestSplits.
            errors (str): see `mldata.TrainTestSplitter.iteritems`.
        Return: instance of ResultsSet.
        """
        results = {}
        for val, split in self.splitter.iteritems(errors):
            logging.info('fit/predict for split {}'.format(val))
            results[val] = self.fit_predict(split)

        # TODO: return RegressionResultsSet instead.
        print(results)
        return ResultsSet(results)


class Results(object):
    """Encapsulate model prediction results & metadata for evaluation."""

    _predicted_suffix = '_predicted'

    def __init__(self, predicted, test_data, fguide, model):
        self.fguide = fguide
        self._pred_colname = '%s%s' % (fguide.target, self._predicted_suffix)
        self.test_data = test_data
        self.test_data.loc[:, self._pred_colname] = predicted
        self.model = model

    @property
    def predicted(self):
        return self.test_data[self._pred_colname]

    @property
    def actual(self):
        return self.test_data[self.fguide.target]

    @property
    def model_params(self):
        """Return model params learned during fitting."""
        return self.model.learned_params

    def save(self, savedir, ow=False):
        """Save the results, including the test data (with predictions in one
        column), the feature guide, and the learned model.

        Args:
            savedir (str): Name of directory to save results to.
            ow (bool): Whether to overwrite the directory if it exists.
        Raises:
            OSError: if savedir exists and ow=False.
        """
        # The Model save method handles creating the directory if it doesn't
        # exist, or raising OSError if it does and ow=False.
        self.model.save(savedir, ow)

        # Save the name of the column with the predicted data.
        # This allows easy reloading.
        self.fguide.real_valueds.add(self._pred_colname)
        self.fguide.save(savedir)

        test_path = os.path.join(savedir, 'test-data.csv')
        mldata.PandasDataset.write_using_fguide(
            self.test_data, test_path, self.fguide)

    @classmethod
    def load(cls, savedir):
        """Load Results from the savedir. Mirrors `Results.save`.

        Args:
            savedir (str): Name of directory to load results from.
        Return:
            results (Results): New instantiation of Results class with loaded
                model, feature guide, and test data (including prediction in
                one column).
        """
        # Read the model.
        model = Model.load(savedir)

        # Read the feature guide.
        fguide_fname = [name for name in os.listdir(savedir)
                        if name.endswith('.conf')][0]
        fguide_path = os.path.join(savedir, fguide_fname)
        fguide = mldata.FeatureGuide(fguide_path)

        # Read the test data.
        test_data = mldata.PandasDataset.read_using_fguide(
            os.path.join(savedir, 'test-data.csv'), fguide)

        try:
            predicted_colname = \
                [name for name in test_data.columns
                 if name.endswith(cls._predicted_suffix)][0]
            predicted = test_data[predicted_colname].values

            # Remove predicted data column name.
            fguide.remove(predicted_colname)
        except IndexError:
            raise ValueError('prediced data not found in test_data file')

        return cls(predicted, test_data, fguide, model)


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


class ResultsSet(Results):
    """Wrap up several related Results objects for aggregate analysis."""

    @staticmethod
    def validate_result_compatibility(results):
        """Ensure the iterable of results can be combined into a ResultsSet.

        There are three criteria for this to work. Each must have:

        1.  same predicted column name
        2.  equivalent feature guides
        3.  same test data columns and dimensions

        The first Result in the iterable is used as the benchmark and compared
        to the others.

        Args:
            results (iterable of Results): Iterable of Results objects.
        Raises:
            ValueError: if there is a mismatch in any of the three necessary
                criteria for compatibility or the iterable is empty.
        """
        # Use the first Result object as the baseline.
        results = list(results)
        try:
            result0 = results[0]
        except IndexError:
            raise ValueError('no results in the iterable')

        # (1) same predicted column name.
        for i, result in enumerate(results):
            if result.predicted.name != result0.predicted.name:
                raise ValueError(
                    'Result %d predicted column name "%s" != "%s"' % (
                        i, result.predicted.name, result0.predicted.name))

        # (2) equivalent feature guides.
        fguide0 = result0.fguide
        for i, result in enumerate(results):
            if result.fguide != fguide0:
                raise ValueError(
                    'Result %d feature guide != Result 0 feature guide' % i)

        # (3) same dimensions and columns for the test data.
        columns0 = np.sort(result0.test_data.columns)
        ncols0 = result0.test_data.shape[1]
        for i, result in enumerate(results):
            if result.test_data.shape[1] != ncols0:
                raise ValueError(
                    '# cols in Result %d test data != # cols in Result 0 test'
                    ' data (%d != %d)' % (
                        i, result.test_data.shape[1], ncols0))
            try:
                columns = np.sort(result.test_data.columns)
                matched = columns0 == columns
                mismatched = not matched.all()
            except ValueError:  # shape mismatch
                mismatched = True
            finally:
                if mismatched:
                    raise ValueError(
                        'Result {} test data columns not equal to'
                        ' Result 0 test data columns: {} != {}'.format(
                            i, np.sort(result.test_data.columns), columns0))

    def __init__(self, results):
        """Initialize the ResultsSet.

        Args:
            results (dict of Results): Results objects that are related. One
                examples would be predictions from a common dataset split on
                different values of the same column. The keys of the dict will
                be used to order the results and to look them up. One choice
                would be the value of the column used to perform trian/test
                splitting. If splits are generated randomly, any ordinals will
                do.
        Raises:
            ValueError: if the results are not compatible (see
                `validate_result_compatibility`).
        """
        # If the results don't match up, we should fail early, so perform
        # necessary sanity checks here.
        self.results = collections.OrderedDict(sorted(results.items()))
        self.validate_result_compatibility(self.results.values())

        self._pred_colname = results[0].predicted.name
        self.fguide = results[0].fguide

    def __getitem__(self, key):
        return self.results[key]

    def __iter__(self):
        return self.results.iterkeys()

    def iteritems(self):
        return self.results.iteritems()

    def iter_results(self):
        return self.results.itervalues()

    @property
    def test_data(self):
        return pd.concat([result.test_data for result in self.results.values()])

    @property
    def predicted(self):
        return self.test_data[self._pred_colname]

    @property
    def actual(self):
        return self.test_data[self.fguide.target]

    @property
    def model_params(self):
        """Return model params learned during fitting."""
        return {key: result.model.learned_params
                for (key, result) in self.results.items()}

    def save(self, savedir, ow=False):
        """Save all results in the savedir, with each individual result saved
        in a subdirectory named using its key.

        Args:
            savedir (str): Name of directory to save results to.
            ow (bool): Whether to overwrite the directory if it exists.
        Raises:
            OSError: if savedir exists and ow=False.
        """
        dirpath = os.path.abspath(savedir)
        logging.info('saving results set to %s' % dirpath)
        saveload.make_or_replace_dir(dirpath, ow)

        # Save each result to a new subdirectory named by its key.
        for key, result in self.results.items():
            path = os.path.join(dirpath, str(key))
            result.save(path, ow)

        # Save the Results class being used for proper reload.
        metadata_file = os.path.join(dirpath, 'metadata.txt')
        with open(metadata_file, 'w') as f:
            f.write(self.results[0].__class__.__name__)

    @classmethod
    def load(cls, savedir):
        """Load Results from the savedir. Mirrors `ResultsSet.save`.

        Args:
            savedir (str): Name of directory to load results from.
        Return:
            resultsSet (ResultsSet): New instantiation of ResultsSet class
                loaded from savedir.
        """
        dirpath = os.path.abspath(savedir)
        logging.info('loading results set from %s' % dirpath)

        # First read metadata. We do this first mostly as an easy way to
        # validate if this directory actually has a ResultsSet saved in it.
        metadata_file = os.path.join(dirpath, 'metadata.txt')
        with open(metadata_file) as f:
            results_class_name = f.read().strip()

        # Look up the actual class object in the module namespace.
        results_class = globals()[results_class_name]

        # Next parse the directory structure to get the Results keys and the
        # directory names where each was saved.
        paths = [os.path.join(dirpath, name) for name in os.listdir(dirpath)]
        subdir_paths = [path for path in paths if os.path.isdir(path)]

        # Now load each Result and instantiate the ResultsSet.
        contents = [(os.path.basename(path), results_class.load(path))
                    for path in subdir_paths]
        results = collections.OrderedDict(sorted(contents))
        return cls(results)
