"""
Tools for loading, preprocessing, formatting, and partitioning csv-formatted
datasets for ML applications. These tools can be used by a variety of models to
quickly convert diverse datasets to appropriate formats for learning.
"""
import os
import logging
import argparse
import itertools

import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from sklearn import preprocessing

from oset import OrderedSet


# Error codes
BOUNDS_FORMAT = 1000
MISSING_ATTRIBUTE = 1001
DIM_MISMATCH = 1002
BAD_FEATURE_CONF = 1003
BAD_FILENAME = 1004
BAD_CMDLINE_ARG = 1005


class BadFeatureConfig(Exception):
    """Raise when bad feature configuration file is found."""
    pass

class FeatureGuide(object):
    """Parse and represent fields of a feature guide."""

    """Guide to feature configuration file field letters."""
    config_guide = {
        't': 'target',
        'i': 'index',
        'k': 'key',
        'e': 'entities',
        'c': 'categoricals',
        'r': 'real_valueds'
    }

    sections = tuple(config_guide.values())
    other_sections = ('target', 'index')
    feature_sections = tuple(set(sections) - set(other_sections))

    # We use slots here not so much for efficiency as for documentation.
    __slots__ = [field_name for field_name in sections]
    __slots__.append('fname')
    __slots__.append('comments')

    @classmethod
    def parse_config(cls, fname):
        """Parse the given configuration file and return a dict of
        {field_letter: list_of_field_names_parsed}.
        """
        with open(fname) as f:
            lines = [l.strip() for l in f.read().split('\n') if l.strip()]

        comments = [l.replace('#', '').strip()
                    for l in lines if l.startswith('#')]
        lines = [l for l in lines if not l.startswith('#')]

        # We use a simple state-machine approach to the parsing
        # in order to deal with multi-line sections.
        parsing = False
        keys = cls.config_guide.keys()
        vars = {var: [] for var in keys}
        for line in lines:
            if not parsing:
                k, csv = line.split(':')
            else:
                csv = line

            vars[k].extend([val.strip() for val in csv.split(',')])
            parsing = not line.endswith(';')
            if not parsing:
                vars[k][-1] = vars[k][-1][:-1]  # remove semi-colon

        # Remove whitespace strings. These may have come from something like:
        # c: this, , that;
        for k in keys:
            vars[k] = [val for val in vars[k] if val]  # already stripped

        return comments, vars

    def restore(self):
        """Parse the configuration and update instance variables to match."""
        self.comments, vars = self.parse_config(self.fname)

        # Sanity checks.
        num_targets = len(vars['t'])
        if num_targets != 1:
            raise BadFeatureConfig(
                'feature config should specify 1 target; got %d; check for'
                'the semi-colon at the end of the t:<target> line' % num_targets)

        num_entity = len(vars['e'])
        if not num_entity:
            raise BadFeatureConfig('0 entity variables given; need at least 1')

        # num_features = len(vars['c']) + len(vars['r'])
        # if not num_features > 0:
        #     raise BadFeatureConfig('no predictors specified')

        # Store extracted field names as instance variables.
        logging.info('read the following feature guide:')
        for k, field_name in self.config_guide.items():
            logging.info('%s: %s' % (field_name, ', '.join(vars[k])))

            # convert to OrderedSet before setting attribute
            setattr(self, field_name, OrderedSet(vars[k]))

        # Extract target variable from its list and store solo.
        self.target = self.target[0]

    def __init__(self, fname):
        """Read the feature guide and parse out the specification.

        The expected file format is the following:

            t:<target>;
            i:<single index field name, if one exists>;
            k:<comma-separated fields that comprise a unique key>;
            e:<comma-separated categorical entity names>;
            c:<comma-separated categorical variable names>;
            r:<comma-separated real-valued variable names>;

        Whitespace is ignored, as are lines that start with a "#" symbol. Any
        variables not included in one of the three groups is ignored. We assume
        the first two categorical variables are the user and item ids.

        Args:
            fname (str): Path of the file containing the feature guide.

        Stores instance variables for each of the field areas under the names in
        FeatureGuide.config_guide.
        """
        self.fname = os.path.abspath(fname)
        self.restore()

    def __repr__(self):
        return '%s("%s")' % (self.__class__.__name__, self.fname)

    def __str__(self):
        return '\n'.join([
            'target: %s' % self.target,
            'index: %s' % ', '.join(self.index),
            'key: %s' % ', '.join(self.key),
            'entities: %s' % ', '.join(self.entities),
            'categoricals: %s' % ', '.join(self.categoricals),
            'real-valueds: %s' % ', '.join(self.real_valueds)
        ])

    @property
    def feature_names(self):
        return list(reduce(
                lambda s1, s2: s1 | s2,
                [getattr(self, attr) for attr in self.feature_sections]))

    @property
    def all_names(self):
        return self.feature_names + [self.target] + list(self.index)

    def remove(self, name):
        """Remove a feature from all sections of the guide where it appears.
        This is disallowed for the target and features in the key.

        Args:
            name (str): The name of the feature to remove.
        Raises:
            AttributeError: if the name is in non-feature sections or the key.
            KeyError: if the name is not in any sections.

        """
        if name in self.other_sections:
            raise AttributeError('cannot remove feature from sections: %s' % (
                ', '.join(self.other_sections)))
        elif name in self.key:
            raise AttributeError('cannot remove features in key')

        # Check all other sections and remove from each where it appears.
        removed = 0
        for section in self.feature_sections:
            try:
                getattr(self, section).remove(name)
                removed += 1
            except KeyError:
                pass

        if not removed:
            raise KeyError("feature '%s' not in feature guide" % name)


"""
A dataset can exist in several different forms. For now, we assume one of two
forms:

    1.  Complete dataset in one file.
    2.  Separate train and test files with identical fields/indices/etc.

We allow the dataset to be initialized from either format and require an
accompanying config file that specifies the fields of the dataset. Since it may
be possible that other forms exist (e.g. with a hold-out set) we use a factory
to load the datasets. By making the first argument to the factory an optional
iterable, we can initialize differently based on the number of filenames
present.
"""
class Dataset(object):
    """Represent a complete dataset existing in one file."""
    pass


class PandasDataset(Dataset):

    @staticmethod
    def index_from_feature_guide(fguide):
        """Return an appropriate index column name based on the feature guide.
        We should use the index as the index if it is present.
        If the index is not present, we should use the key as the index only
        if none of those fields are also used as features.
        If some of those fields are used as features, we should simply make a
        new index and add it to the feature guide; in this case, return None.
        """
        if fguide.index:
            return list(fguide.index)
        elif (fguide.key and
              (fguide.key - fguide.feature_names) == fguide.key):
                return list(fguide.key)
        else:
            return None

    def index_colname(self):
        return self.index_from_feature_guide(self.fguide)

    def __init__(self, fname, config_file):
        """Load the feature configuration and then load the columns present in
        the feature config.
        """
        self.fguide = fguide = FeatureGuide(config_file)
        self.fname = os.path.abspath(fname)

        # We only need to load the columns that show up in the config file.
        index_col = self.index_colname()
        # Note that pandas interprets index_col=None as "make new index."
        self.dataset = pd.read_csv(
            fname, usecols=fguide.all_names, index_col=index_col)

        # Instance variables to store metadata generated during transformations.
        self.column_maps = {}  # mapping from one space to another
        self.imputations = {}  # imputing missing values
        self.scalers = {}      # scaling column values

    @property
    def reals(self):
        return self.dataset[list(self.fguide.real_valueds)]

    @property
    def categoricals(self):
        return self.dataset[list(self.fguide.categoricals)]

    @property
    def entities(self):
        return self.dataset[list(self.fguide.entities)]

    @property
    def key(self):
        return self.dataset[list(self.fguide.key)]

    def map_column_to_index(self, col):
        """Map values in column to a 0-contiguous index. This enables use of
        these attributes as indices into an array (for bias terms, for
        instance). This method changes the ids in place, producing an (new_id,
        orig_id) dict which is stored in the `column_maps` instance variable.

        Args:
            key (str): Column name with ids to map.
        """
        # First construct the map from original ids to new ones.
        ids = self.dataset[col].unique()
        n = len(ids)
        idmap = dict(itertools.izip(ids, xrange(n)))

        # Next use the map to conver the ids in-place.
        self.dataset[col] = self.dataset[col].apply(lambda _id: idmap[_id])

        # Now swap key for value in the idmap to provide a way to convert back.
        self.column_maps[col] = {val: key for key, val in idmap.iteritems()}

    def remove_feature(self, name):
        """Remove the given feature from the feature guide and then from the
        dataset.

        Args:
            name (str): Name of the feature to remove.
        Raises:
            AttributeError: if the name is in non-feature sections or the key
                section of the feature guide.
            KeyError: if the name is not in feature guide or not in the dataset.
        """
        self.fguide.remove(name)
        self.dataset = self.dataset.drop(name, axis=1)

    def column_is_all_null(self, column):
        return self.dataset[column].isnull().sum() == len(self.dataset)

    def verify_columns_in_dataset(self, columns):
        """Ensure all columns are present in the dataset before doing some
        operation to avoid side effects or the need for rollback.
        """
        all_cols = self.dataset.columns
        for col in columns:
            if not col in all_cols:
                raise KeyError("column '%s' not in dataset" % col)

    def impute(self, columns, method='median', all_null='raise'):
        """Perform missing value imputation for the given columns using the
        specified `pandas.DataFrame` method for the fill value. All NaN values
        in the columns will be replaced with this value.

        Args:
            columns (iterable of str): Column names to perform missing value
                imputation on.
            method (str): Name of the `pandas.DataFrame` method to use to
                compute the fill value.
            all_null (str): One of {'drop', 'raise', 'ignore'}, this defines the
                action taken when a column with only missing values is
                encountered. If drop, the entire column is dropped. If raise, a
                ValueError is raised. If ignore, the column is ignored.
        Raises:
            KeyError: if any of the column names are not in the dataset.
            ValueError: if 'raises is specified for `all_null` and an all null
                column is encountered.

        """
        # Ensure all_null is one of the valid choices.
        allowed = {'drop', 'raise', 'ignore'}
        if all_null not in allowed:
            raise ValueError(
                'all_null must be one of: %s' % ', '.join(allowed))

        self.verify_columns_in_dataset(columns)

        # If all_null='raise', check all columns first to avoid side effects.
        if all_null == 'raise':
            for col in columns:
                if self.column_is_all_null(col):
                    raise ValueError("all null column '%s'" % col)

        for col in columns:
            if self.column_is_all_null(col):
                if all_null == 'drop':
                    self.remove_feature(col)
                    logging.info("all null column '%s' was dropped" % col)
                    continue
                # Already checked all_null == 'raise'
                else:
                    logging.info("all null column '%s' ignored" % col)

            # Compute fill value and fill all NaN values.
            column = self.dataset[col]
            fill_value = getattr(column, method)()
            self.dataset[col] = column.fillna(fill_value)

            # Store fill_value imputed.
            self.imputations[col] = fill_value

    def impute_reals(self):
        self.impute(self.fguide.real_valueds)

    def scale(self, columns):
        """Z-score scale the given columns IN-PLACE, storing the scalers used in
        the `scalers` instance variable. The scaling can be reversed using
        `unscale`.

        Args:
            columns (iterable of str): Column names to scale.
        Raises:
            KeyError: if any of the column names are not in the dataset.

        """
        self.verify_columns_in_dataset(columns)

        for col in columns:
            # First ensure the column has not already been scaled.
            if col in self.scalers:
                scaler, scaled = self.scalers[col]
            else:
                scaler = preprocessing.StandardScaler()
                scaled = False

            if not scaled:
                self.scalers[col] = (scaler, True)
                self.dataset[col] = scaler.fit_transform(self.dataset[col])

    def scale_reals(self):
        if self.fguide.real_valueds:
            self.scale(self.fguide.real_valueds)

    def unscale(self, columns):
        """Reverse the Z-score scaling on the given columns IN-PLACE. If any of
        the given columns have not been scaled, they are ignored.

        Args:
            columns (iterable of str): Column names to unscale.
        Raises:
            KeyError: if any of the column names are not in the dataset.

        """
        self.verify_columns_in_dataset(columns)

        for col in columns:
            scaler, scaled = self.scalers.get(col, (0, 0))
            if not scaled:
                logging.info("column '%s' has not been scaled, ignoring" % col)
                continue

            self.dataset[col] = scaler.inverse_transform(self.dataset[col])
            self.scalers[col][1] = False

    # TODO: Incorporate preprocess as a method.
    def preprocess(self, impute=True):
        """Return preprocessed (X, y, eid) pairs for the train and test sets.

        Preprocessing includes:

        1.  Map all entity IDs to a 0-contiguous range.
        2.  Z-score scale the real-valued features.
        3.  One-hot encode the categorical features (including entity IDs).

        This function tries to be as general as possible to accomodate learning by
        many models. As such, there are a variety of return values:

        1.  eids: entity IDs as a numpy ndarray
        2.  X: feature vector matrix (first categorical, then real-valued)
        3.  y: target values (unchanged from input)
        4.  indices: The indices of each feature in the encoded X matrix.
        5.  nents: The number of unique entities for each entity.

        """
        eids = {}
        for entity in self.fguide.entities:
            self.map_column_to_index(entity)
            eids[entity] = self.dataset[entity].values

        # Z-score scaling of real-valued features.
        self.impute_reals()
        self.scale_reals()
        nreal = len(self.fguide.real_valueds)

        # One-hot encoding of entity and categorical features.
        cats_and_ents = list(self.fguide.entities | self.fguide.categoricals)
        all_cats = self.dataset[cats_and_ents]
        encoder = preprocessing.OneHotEncoder()
        encoded_cats = encoder.fit_transform(all_cats)

        # Create a feature map for decoding one-hot encoding.
        ncats_and_ents = encoder.active_features_.shape[0]
        nf = ncats_and_ents + nreal

        # Count entities.
        logging.info('after one-hot encoding, found # unique values:')
        counts = np.array([
            all_cats[cats_and_ents[i]].unique().shape[0]
            for i in range(len(cats_and_ents))
        ])

        # Assemble map to new feature indices using counts.
        indices = zip(cats_and_ents, np.cumsum(counts))
        for attr, n_values in zip(self.fguide.entities, counts):
            logging.info('%s: %d' % (attr, n_values))

        # Add in real-valued feature indices.
        last_cat_index = indices[-1][1]
        indices += zip(self.fguide.real_valueds,
                       range(last_cat_index + 1, nf + 1))

        # How many entity and categorical features do we have now?
        nents = dict(indices)[self.fguide.entities[-1]]
        ncats = ncats_and_ents - nents
        nf = ncats_and_ents + nreal

        n_ent_names = len(self.fguide.entities)
        ent_idx = range(n_ent_names)
        cat_idx = range(n_ent_names, len(cats_and_ents))
        nactive_ents = sum(encoder.n_values_[i] for i in ent_idx)
        nactive_cats = sum(encoder.n_values_[i] for i in cat_idx)

        logging.info('number of active entity features: %d of %d' % (
            nents, nactive_ents))
        logging.info('number of active categorical features: %d of %d' % (
            ncats, nactive_cats))
        logging.info('number of real-valued features: %d' % nreal)
        logging.info('Total of %d features after encoding' % nf)

        # Put all features together.
        X = sp.sparse.hstack((encoded_cats, self.reals))\
            if nreal else encoded_cats
        y = self.dataset[self.fguide.target].values
        return X, y, eids, indices, nents

    # TODO: Take the code from scaffold.py and incorporate. This includes logic
    # for splitting the train and test data. This logic is somewhat general --
    # splitting a dataset based on a time attribute, so you can write it more
    # generally in that line of thinking. You can make it even more general:
    # splitting based on a binary comparison relationship with one of the
    # columns. You could also have a method which builds on the simple split to
    # perform sequential splits based on all unique values of a column, applying
    # one comparison to get the train set and another to get the test set.


def read_train_test(train_file, test_file, conf_file):
    """Read the train and test data according to the feature guide in the
    configuration file. Return the train and test data as (X, y) pairs for the
    train and test data.

    Args:
        train_file (str): Path of the CSV train data file.
        test_file (str): Path of the CSV test data file.
        conf_file (str): Path of the configuration file.

    Returns:
        tuple: (train_eids, train_X, train_y,
                test_eids, test_X, test_y,
                feat_indices, number_of_entities),
               preprocessed train/test data and mappings from features to
               indices in the resulting data matrices. Both train and test are
               accompanied by 0-contiguous primary entity id arrays.

    """
    try:
        target, ents, cats, reals = read_feature_guide(conf_file)
    except IOError:
        raise IOError('invalid feature guide conf file path: %s' % conf_file)

    to_read = [target] + ents + cats + reals
    def read_file(name, fpath):
        try:
            data = pd.read_csv(fpath, usecols=to_read)
        except IOError:
            raise IOError('invalid %s file path: %s' % (name, fpath))
        except ValueError as err:
            attr_name = err.args[0].split("'")[1]
            attr_type = ('entity' if attr_name in ents else
                         'categorical' if attr_name in cats else
                         'real-valued' if attr_name in reals else
                         'target')
            raise BadFeatureConfig('%s attribute %s not found in %s file' % (
                attr_type, attr_name, name))

        return data

    train = read_file('train', train_file)
    test = read_file('test', test_file)
    n_train = train.shape[0]
    n_test = test.shape[0]
    logging.info('number of instances: train=%d, test=%d' % (
        train.shape[0], test.shape[0]))

    return train, test, target, ents, cats, reals


class Model(object):
    """General model class with load/save/preprocess functionality."""

    def set_fguide(self, fguidef=''):
        if fguidef:
            self.read_fguide(fguidef)
        else:
            self.reset_fguide()

    def read_fguide(self, fguidef):
        self.target, self.ents, self.cats, self.reals = \
            read_feature_guide(fguidef)

    def reset_fguide(self):
        self.target = ''
        for name in ['ents', 'cats', 'reals']:
            setattr(self, name, [])

    @property
    def fgroups(self):
        return ('ents', 'cats', 'reals')

    @property
    def nf(self):
        return sum(len(group) for group in self.fgroups)

    def check_fguide(self):
        if self.nf <= 0:
            raise ValueError("preprocessing requires feature guide")

    def read_necessary_fguide(self, fguide=''):
        if fguide:
            self.read_fguide(fguide)
        else:
            self.check_fguide()

    def preprocess(self, train, test, fguidef=''):
        self.read_necessary_fguide(fguidef)
        return preprocess(
            train, test, self.target, self.ents, self.cats, self.reals)

    def check_if_learned(self):
        unlearned = [attr for attr, val in self.model.items() if val is None]
        if len(unlearned) > 0 or len(self.model) == 0:
            raise UnlearnedModel("IPR predict with unlearned params", unlearned)

    def save(self, savedir, ow=False):
        save_model_vars(self.model, savedir, ow)

    def load(self, savedir):
        self.model = load_model_vars(savedir)
