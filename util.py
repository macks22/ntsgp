"""
Useful utility functions.

"""
import luigi
import luigi.worker


def schedule_task(task, verbose=False):
    if verbose:
        luigi.interface.setup_interface_logging()
    sch = luigi.scheduler.CentralPlannerScheduler()
    w = luigi.worker.Worker(scheduler=sch)
    w.add(task)
    w.run()


def abbrev_names(names, nletters=3):
    """Abbreviate colnames using as few letters as possible and combine them
    using title casing, with 0 spaces in between. For terms with gooseneck
    naming, the first letter of each term up to the last is used (in caps), and
    the last first few letters are taken from the last name. So for instance,
    colnames=('nation_zip', 'nation_code', 'name', 'cum_gpa', 'term_gpa') would
    give: ZiCoNaCuTodNaCGpaTGpa. We use 2 letters minimum from each colname,
    unless it only has one.
    """
    if not names:
        return ''

    # sort names
    names = list(sorted(names))

    # first break up the gooseneck names.
    abbrevs = []
    terms = [name.split('_') for name in names]
    end = nletters + 1  # use to end ranges
    for words in terms:
        if len(words) == 1:
            abbrev = words[0][:nletters]
            abbrev = ''.join([abbrev[0].upper(), abbrev[1:end]])
        else:
            last_term = words[-1]
            prefix = ''.join([word[0] for word in words[:-1]]).upper()
            abbrev = ''.join([prefix, last_term[0].upper(), last_term[1:end]])
        abbrevs.append(abbrev)

    return ''.join(abbrevs)


def fname_from_cname(cname):
    words = []
    chars = [cname[0]]
    for c in cname[1:]:
        if c.isupper():
            words.append(''.join(chars))
            chars = [c]
        else:
            chars.append(c)

    words.append(''.join(chars))
    return '-'.join(map(lambda s: s.lower(), words))


