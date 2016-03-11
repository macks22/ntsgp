"""
Functionality for file naming, useful for saving datasets, splits, model
parameters, model results, etc.

"""

import itertools

from oset import OrderedSet


def abbreviate_name_firstn(name, first_n=1):
    """Abbreviate a "_"-separated name by keeping the first_n letter of
    each word after splitting on underscores.

    Args:
        name (str): An underscore-separated name to abbreviate.
        first_n (int): How many letters to keep from each word.

    """
    return ''.join([word[:first_n] for word in name.split('_')])


def abbreviate_name_cyclic(name):
    """Return a generator that yields successively longer (and more unique)
    abbrevations for a "_"-separated name. It starts by abbreviating as the
    first letter of each word after splitting on underscores. Next, it
    tacks on the second letter of the first word, followed by second of the
    second, and so on, then third letter, etc. until all letters are
    included. The underscores remain absent even if all letters are
    included.

    Args:
        name (str): An underscore-separated name to abbreviate.
    Return:
        abbrevs (generator of str): A generator yielding the abbreviated
            versions of the name.
    """
    words = name.split('_')
    chars = filter(lambda char: char is not None,
                reduce(lambda l1, l2: l1 + l2,
                    itertools.izip_longest(*words)))

    n_words = len(words)
    n_chars = len(chars)
    for n in range(n_words, n_chars + 1):
        yield ''.join(chars[:n])


def abbreviate_name_upton(name, upton=1):
    """Abbreviate "_"-separated name using first n letters after removing
    underscores. The letters are drawn such that each word remains in order
    and the letters in each word remain in order. For instance,

    >>> abbreviate_name_upton('max_leaf_nodes', 4)
    'maln'
    >>> abbreviate_name_upton('max_leaf_nodes', 5)
    'malen'
    >>> abbreviate_name_upton('max_leaf_nodes', 7)
    'maxleno'

    Args:
        name (str): An underscore-separated name to abbreviate.

    """
    words = name.split('_')
    n_words = len(words)
    chars_per_word = [upton / n_words] * n_words
    remaining = upton % n_words
    for i in range(remaining):
        chars_per_word[i] += 1

    return ''.join([words[i][:chars_per_word[i]] for i in range(n_words)])


def abbreviate_value(value, decimals=4, first_n=3):
    if isinstance(value, float):
        fmt_str = '%.{}f'.format(decimals)
        return (fmt_str % value).replace('.', '_')
    elif isinstance(value, int):
        return '%d' % value
    elif isinstance(value, basestring):
        if filter(lambda c: c.isalpha(), value).isupper():
            return value  # probably acronym
        else:
            return abbreviate_name_firstn(value, first_n)
    elif isinstance(value, bool):
        return '%d' % int(value)
    else:  # some object
        try:
            return value.__name__
        except AttributeError:
            return value.__class__.__name__


def abbreviate_names_uniquely(names):
    """Given a list of strings, return a list of strings with abbreviations
    for each that are unique among the strings given. Strings occuring
    later in the iterable will have longer names if there are conflicts
    with shorter abbreviations. If an unordered iterable is passed, the
    abbreviations assigned may be different each time. So this is really
    only meant to be used with ordered iterables.

    Args:
        names (iterable of str): Iterable of names to abbreviate.
    Return:
        abbrevs (dict): Mapping from names given to their abbreviations.

    """
    abbrevs = {}
    unique = OrderedSet()
    for name in names:
        words = name.split('_')
        n_words = len(words)
        abbrev = abbreviate_name_upton(name, n_words)

        # Sometimest last character is a parameter distinguisher, such as in
        # theta0, theta1, theta2, or as in lambda_B, lambda_W. The second case
        # falls under the typical use case, so should be accounted for to avoid
        # repeated last characters. The first case is atypical; without an
        # underscore preceding it, the last character is normally be discarded.
        last_char = name[-1]
        last_char_important = last_char.isupper() or last_char.isdigit()
        last_char_not_last_word = len(words[-1]) > 1
        add_last_char = last_char_important and last_char_not_last_word

        if add_last_char:
            abbrev += last_char

        while abbrev in unique:
            n_words += 1
            abbrev = abbreviate_name_upton(name, n_words)
            if add_last_char:
                abbrev += last_char

        unique.add(abbrev)
        abbrevs[name] = abbrev

    return abbrevs


def suffix_from_params(param_dict):
    """All scikit-learn estimators have a `get_params` method which returns a
    dictionary of the parameter names associated with their values. Parameters
    not set are assigned to None.

    This function returns a hyphen-separated abbrevation of the parameter
    settings for use in file names.

    Args:
        param_dict (dict): Map from parameter names to values.
    Return:
        abbrev (str): Hyphen-separated string of parameter name@value
            abbreviations.
    """
    non_null = [(k, v) for k, v in param_dict.items() if v is not None]

    # Get abbreviations for all names to avoid conflict across runs. For
    # instance, consider two params `random_start` and `random_state`. If one
    # is set in the first run and the second is None, it would get the "rs"
    # abbreviation. If the second was set in th second run and the first was
    # None, then we would still see "rs" but it would now represent the other
    # parameter. So we get a unique abbreviation across all names
    # alphabetically and use that to avoid this type of conflict.
    all_names = list(sorted(param_dict.keys()))
    abbrevs = abbreviate_names_uniquely(all_names)
    pairs = [(abbrevs[name], abbreviate_value(val)) for name, val in non_null]
    return '-'.join(['%s@%s' % (k, v) for k, v in sorted(pairs)])
