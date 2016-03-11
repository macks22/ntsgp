"""
Model and Results classes for encapsualting ML method train and predict logic
as well as prediction evaluation logic.
"""
import os
import abc
import copy
import json
import pydoc
import inspect
import logging
import warnings
import importlib
import collections
import multiprocessing as mp

import numpy as np
import pandas as pd

import mldata
import naming
import saveload


class Model(object):
    """Encapsulate model with known API for TrainTestSplit use."""

    def __init__(self, model, normalize=True, use_cats=True, ohc_cats=True,
                 use_ents=True, ohc_ents=True, remove_cold_start=True):
        """Take the model and several arguments that specify which preprocessing
        steps should be used for this model. The model should be an sklearn
        compliant estimator. For the rest of the arguments, see
        `PandasTrainTestSplit.preprocess`.
        """
        self.model = model
        self.normalize = normalize
        self.use_cats = use_cats
        self.ohc_cats = ohc_cats
        self.use_ents = use_ents
        self.ohc_ents = ohc_ents
        self.remove_cold_start = remove_cold_start

    @property
    def preprocess_args(self):
        return {name: getattr(self, name) for name in
                ('normalize', 'use_cats', 'ohc_cats', 'use_ents', 'ohc_ents',
                 'remove_cold_start')}

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

        model = self.__class__(inner_model, **self.preprocess_args)
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
        accepted = self.fit_kwargs + self.fit_pargs
        filtered_kwargs = key_intersect(kwargs, accepted)
        self.model.fit(X, np.squeeze(y), **filtered_kwargs)
        return self

    def predict(self, X, **kwargs):
        """Predict new target variables using the fitted model. This method
        uses introspection to determine which keyword arguments are accepted
        and only passes those through.
        """
        accepted = self.predict_kwargs + self.predict_pargs
        filtered_kwargs = key_intersect(kwargs, accepted)
        return self.model.predict(X, **filtered_kwargs)


class SklearnModelMP(SklearnModel, mp.Process):
    """Multiprocessing variant of SklearnModel."""

    def __init__(self, model, split, pipe, *args, **kwargs):
        """Takes a model, a TrainTestSplit, and a pipe connected to the caller.

        Args:
            model (sklearn.estimator): A model with the scikit-learn estimator
                interface. This will be fitted to the training set and used to
                make predictions on the test set.
            split (TrainTestSplit): The training and test data.
            pipe (multiprocessing.Pipe): For communication to the parent
                process -- to communicate results and learned parameters.
        """
        SklearnModel.__init__(self, model)
        mp.Process.__init__(self, *args, **kwargs)
        self.split = split
        self.pipe = pipe

    def run(self):
        # Create copy of model with same params.
        model = self.model.clone()

        train_X, train_y, train_eids,\
        test_X, test_y, test_eids, fmap, nents = \
            split.preprocess(all_null='drop', **model.preprocess_args)

        # Extraneous kwargs are filtered by the fit/predict methods.
        kwargs = {'entity_ids': train_eids,
                  'feature_indices': indices,
                  'n_entities': nents}

        self.fit(train_X, train_y, **kwargs)
        pred_y = self.predict(test_X, **kwargs)
        self.pipe.send([pred_y, self.split.test, self.split.fguide, model])
        return 0


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
        # Create copy of model with same params.
        model = self.model.clone()

        train_X, train_y, train_eids,\
        test_X, test_y, test_eids, fmap, nents = \
            split.preprocess(all_null='drop', **model.preprocess_args)

        # Extraneous kwargs are filtered by the fit/predict methods.
        kwargs = {'entity_ids': train_eids.values,
                  'feature_indices': fmap,
                  'n_entities': nents}
        model.fit(train_X, train_y, **kwargs)

        kwargs['entity_ids'] = test_eids.values
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
        return self.fit_predict(split)

    def fit_predict_all(self, errors='log', parallel=True):
        """Run sequential fit/predict loop for all possible data splits in a
        generative manner.

        Outstanding TODOs:

        1.  results not fitting in memory
        2.  optional reassembly of full data frame with original and predicted
            results.

        Args:
            parallel (bool): Run the separate splits using multiple processes
                if True, else just use single main process. True by default.
            errors (str): see `mldata.TrainTestSplitter.iteritems`.
        Return: instance of ResultsSet.
        """
        if parallel:
            return self._fit_predict_all_parallel(errors)
        else:
            return self._fit_predict_all(errors)

    def _fit_predict_all(self, errors='log'):
        """Run sequential fit/predict loop for all possible data splits in a
        generative manner.

        Args:
            errors (str): see `mldata.TrainTestSplitter.iteritems`.
        Return: instance of ResultsSet.
        """
        results = {}
        for val, split in self.splitter.iteritems(errors):
            logging.info('fit/predict for split {}'.format(val))
            results[val] = self.fit_predict(split)

        return RegressionResultsSet(results)

    def _fit_predict_all_parallel(self, errors='log'):
        """Parallel variant of fit_predict_all."""
        results = {}
        for key, split in self.splitter.iteritems(errors):
            parent_conn, child_conn = mp.Pipe()
            proc = SklearnModelMP(self.model.clone().model, split, child_conn)
            results[key] = (proc, parent_conn)

            logging.info(
                'starting process for fit/predict on split {}'.format(key))
            proc.start()

        for key, (proc, conn) in results.items():
            pred_y, test, fguide, inner_model = conn.recv()
            model = SklearnModel(inner_model)
            results[key] = RegressionResults(pred_y, test, fguide, model)

        return RegressionResultsSet(results)


class abstractclassmethod(classmethod):

    __isabstractmethod__ = True

    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super(abstractclassmethod, self).__init__(callable)


class ResultsBase(object):
    """Abstract base class for prediction results on a TrainTestSplit."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def predicted(self):
        pass

    @abc.abstractproperty
    def actual(self):
        pass

    @abc.abstractproperty
    def model_params(self):
        pass

    @abc.abstractmethod
    def save(self, savedir, ow=False):
        pass

    @abstractclassmethod
    def load(self, savedir):
        pass


class Results(ResultsBase):
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


# Define regression metrics that take an array of errors.
def error_rmse(arr):
    return np.sqrt((arr ** 2).sum() / len(arr))

def error_mae(arr):
    return abs(arr).sum() / len(arr)

def error_mae_std(arr):
    return abs(arr).std()


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

    def evaluate_by(self, colname,
                    metrics=[error_rmse, error_mae, error_mae_std, len]):
        """Evaluate the results in terms of (grouping by) a particular column.

        Args:
            colname (str): Name of column to evaluate by.
            metrics (iterable): Metrics to apply to the grouped error.
        Return:
            eval (DataFream): An evaluation of the results by the given column
                name, including various metrics: RMSE, MAE, MAE std, counts.
        """
        errname = '__error__'
        test_data = self.test_data
        test_data.loc[:, errname] = self.error()
        evaluation = test_data.groupby(colname)[errname].aggregate(metrics)
        evaluation = pd.concat((evaluation, \
            test_data.groupby(lambda i: 'all')[errname].agg(metrics)))
        return evaluation


class ColumnMismatchError(Exception):
    """Raise when aggregating DataFrame objects with mismatched columns."""
    pass

class ResultsSet(ResultsBase):
    """Wrap up several related Results objects for aggregate analysis."""

    @staticmethod
    def _get_first(results):
        try:
            return results.iteritems().next()
        except StopIteration:
            raise ValueError('no results in the iterable')

    @staticmethod
    def verify_columns_match(results):
        """Verify test data for each result has matching columns.

        Args:
            results (OrderedDict of Results): To verify columns from.
        Raises:
            ValueError: if the iterable is empty or any columns do not match.
        """
        # Use the first Result object as the baseline.
        first_key, first_result = ResultsSet._get_first(results)

        columns0 = np.sort(first_result.test_data.columns)
        ncols0 = first_result.test_data.shape[1]
        for key, result in results.iteritems():
            ncols = result.test_data.shape[1]
            if ncols != ncols0:
                raise ColumnMismatchError(
                    '# cols in Result {} test data != # cols in Result {} test'
                    ' data ({} != {})'.format(key, first_key, ncols, ncols0))
            try:
                columns = np.sort(result.test_data.columns)
                matched = columns0 == columns
                mismatched = not matched.all()
            except ValueError:  # shape mismatch
                mismatched = True
            finally:
                if mismatched:
                    raise ColumnMismatchError(
                        'Result {} test data columns not equal to Result {}'
                        ' test data columns: {} != {}'.format(
                            key, first_key, columns, columns0))

    @staticmethod
    def verify_same_predicted_name(results):
        """Ensure all Results have the same column name for predictions.
        Args:
            results (OrderedDict of Results): To check compatibility of.
        Raises:
            ValueError: if any predictions do not match.
        """
        # Use the first Result object as the baseline.
        first_key, first_result = ResultsSet._get_first(results)

        # (1) same predicted column name.
        for key, result in results.iteritems():
            name = result.predicted.name
            if name != first_result.predicted.name:
                raise ValueError(
                    'Result {} predicted column name "{}" != "{}"'.format(
                        key, name, first_result.predicted.name))

    @staticmethod
    def verify_same_feature_guide(results):
        """Verify the Results have equivalent feature guides.
        Args:
            results (OrderedDict of Results): To check compatibility of.
        Raises:
            ValueError: if any of the feature guides do not match.
        """
        # Use the first Result object as the baseline.
        first_key, first_result = ResultsSet._get_first(results)

        first_fguide = first_result.fguide
        for key, result in results.iteritems():
            if result.fguide != first_fguide:
                raise ValueError(
                    'Result {} feature guide != Result {} feature'
                    ' guide'.format(key, first_key))

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
            results (OrderedDict of Results): To check compatibility of.
        Raises:
            ValueError: if there is a mismatch in any of the three necessary
                criteria for compatibility or the iterable is empty.
        """
        ResultsSet.verify_same_predicted_name(results)
        ResultsSet.verify_same_feature_guide(results)
        ResultsSet.verify_columns_match(results)

    def __init__(self, results, col_mismatch='fill'):
        """Initialize the ResultsSet.

        Args:
            results (dict of Results): Results objects that are related. One
                examples would be predictions from a common dataset split on
                different values of the same column. The keys of the dict will
                be used to order the results and to look them up. One choice
                would be the value of the column used to perform trian/test
                splitting. If splits are generated randomly, any ordinals will
                do.
            col_mismatch (str): One of {'fill', 'raise'}, this specifies the
                action to take if the columns don't match up between all the
                Result objects test data frames. If 'fill', simply add the
                column into the test data frames where it is missing with NaN
                values. If 'raise', raise a ValueError indicating the mismatch.
        Raises:
            ValueError: if the results are not compatible (see
                `validate_result_compatibility`).
        """
        results = collections.OrderedDict(sorted(results.items()))
        results_list = results.values()
        result0 = results_list[0]

        # If the results don't match up, we should fail early, so perform
        # necessary sanity checks here.
        try:
            self.verify_columns_match(results)

            # no mismatch, can simply use the first feature guide.
            self.fguide = result0.fguide
            self.verify_same_feature_guide(results)
        except ColumnMismatchError:
            if col_mismatch == 'raise':
                raise

            # else fill using an outer join, which automatically replaces all
            # missing columns in any DataFrame with NaN columns.
            test_data_frames = [res.test_data for res in results_list]
            concat = pd.concat(test_data_frames, join='outer')

            # Now pull the individual frames back out.
            start = 0
            for key in results:
                df = results[key].test_data
                nsamples = df.shape[0]
                end = start + nsamples
                results[key].test_data = concat[start:end].copy()
                start = end

            # And union all feature guides to get the representative one.
            self.fguide = reduce(
                lambda fg1, fg2: fg1.union(fg2),
                [result.fguide for result in results_list])

        # Assign all individual results to have same, shared representative
        # feature guide.
        for result in results_list:
            result.fguide = self.fguide

        self.verify_same_predicted_name(results)
        self._pred_colname = result0.predicted.name

        # All good, go ahead and set instance variables.
        self.results = results

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

        # Save metadata necessary for proper reload.
        metadata_file = os.path.join(dirpath, 'metadata.json')
        first_key, first_result = self.results.iteritems().next()

        key_type = type(first_key)
        if hasattr(key_type, 'item'):  # convert numpy types to native
            key_type = type(first_key.item())
        else:
            key_type = type(first_key)

        metadata = {
            'key_type': key_type.__name__,  # assume all keys are same type
            'results_class': first_result.__class__.__name__
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)

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
        metadata_file = os.path.join(dirpath, 'metadata.json')
        with open(metadata_file) as f:
            metadata = json.load(f)

        # Extract metadata.
        key_type = pydoc.locate(metadata['key_type'])
        results_class_name = metadata['results_class']

        # Look up the actual class object in the module namespace.
        results_class = globals()[results_class_name]

        # Next parse the directory structure to get the Results keys and the
        # directory names where each was saved.
        paths = [os.path.join(dirpath, name) for name in os.listdir(dirpath)]
        subdir_paths = [path for path in paths if os.path.isdir(path)]

        # Now load each Result and instantiate the ResultsSet, converting the
        # keys to the appropriate types (from the metadata file).
        contents = [(key_type(os.path.basename(path)),
                     results_class.load(path))
                    for path in subdir_paths]
        results = collections.OrderedDict(sorted(contents))
        return cls(results)


class RegressionResultsSet(ResultsSet, RegressionResults):
    """A ResultsSet with evaluation metrics for regression predictions."""
    pass

