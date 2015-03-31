"""
Construct a major flow diagram.
"""
import pandas as pd
import numpy as np
import igraph


COLLEGES = [
    ['#1f77b4', 'I', 'Incoming Freshman'],
    ['#2ca02c', 'G', 'Graduate'],
    ['#7f7f7f', 'D', 'Drop Out'],
    ['#8c564b', 'UN', 'University (Provost)'],
    ['#9edae5', 'LA', 'Humanities & Social Sciences'],
    ['#17becf', 'SC', 'College of Science'],
    ['#98df8a', 'BU', 'School of Management'],
    ['#9467bd', 'VS', 'Volgenau School of Engr.'],
    ['#aec7e8', 'AR', 'Visual & Performing Arts'],
    ['#c5b0d5', 'HH', 'Health and Human Services'],
    ['#c49c94', 'E1', 'Education & Human Development'],
    ['#d62728', 'CA', 'Conflict Analysis & Resolution'],
    ['#e377c2', 'PP', 'School of Policy, Government and International Affairs'],
    ['#f7b6d2', 'LW', 'School of Law']
]

COLORS = [
    '#1f77b4',
    '#2ca02c',
    '#7f7f7f',
    '#8c564b',
    '#9edae5',
    '#17becf',
    '#98df8a',
    '#9467bd',
    '#aec7e8',
    '#c5b0d5',
    '#c49c94',
    '#d62728',
    '#e377c2',
    '#f7b6d2',
    '#ff7f0e',
    '#ff9896',
    '#ffbb78'
]


def read_student_df(fname='nsf_student.csv'):
    student_df = pd.read_csv(fname)

    # Build a MultiIndex
    tuples = list(zip(student_df.TERMBNR.values, student_df.id.values))
    idx = pd.MultiIndex.from_tuples(tuples)

    # Then reindex the student data frame using that index
    sframe = pd.DataFrame(student_df.values, columns=student_df.columns,
                          index=idx)

    # Get all unique student ids and sort them, to build an idmap
    student_ids = student_df['id'].unique()
    student_ids.sort()
    idmap = {sid: idx for idx, sid in np.ndenumerate(student_ids)}

    # Now add the mapping to the student data frame
    mapping = pd.DataFrame.from_dict(idmap, orient='index')
    mapping.columns = ('idx',)
    students = pd.merge(sframe, mapping, left_on=('id',), right_index=True)
    # del students['id']
    del students['TERMBNR']

    # remove duplicate indices, keeping first occurence
    students = students.groupby(level=(0,1)).first()
    return students


def label_students(student_df):
    # Get all unique student ids and sort them, to build an idmap
    student_ids = student_df['id'].unique()
    student_ids.sort()

    # Build a label matrix for the 16 terms + 1
    terms = student_df['TERMBNR'].unique()
    terms = [200910] + list(terms)
    terms.sort()
    num_labels = len(terms)
    slabels = np.ndarray((len(student_ids), num_labels), dtype=np.object_)
    slabels.fill('N')  # not attending

    for term_num, term in enumerate(terms):
        for sid in student_ids:
            try:
                record = student_df.ix[term].ix[sid]
                coll = record['PCOLL']
                idx = record['idx']
                slabels[idx][term_num] = coll
            except KeyError:  # no courses in term
                pass

    # Put labels in data frame and annotate with terms as columns and student
    # ids as index
    labeldf = pd.DataFrame(slabels)
    labeldf.columns = terms
    labeldf.index = student_ids
    labeldf.to_csv('student-college-labels.csv')
    return labeldf


def next_term(termbnr):
    """Return the next term as an int."""
    term = str(termbnr)
    year = int(term[:4])
    season = term[4:]
    if season == '70':
        tstring = "%d%d" % (year+1, 10)
    elif season == '40':
        tstring = "%d%d" % (year, 70)
    else:
        tstring = "%d%d" % (year, 40)
    return int(tstring)


def prev_term(termbnr):
    """Return the previous term as an int."""
    term = str(termbnr)
    year = int(term[:4])
    season = term[4:]
    if season == '10':
        tstring = "%d%d" % (year-1, 70)
    elif season == '40':
        tstring = "%d%d" % (year, 10)
    else:
        tstring = "%d%d" % (year, 40)
    return int(tstring)


def add_admissions_labels(labeldf, fname='nsf_admissions.csv'):
    admiss = pd.read_csv(fname)
    aterms = admiss.drop_duplicates('id')[['id', 'cohort', 'Application_College']]
    aterms['id'] = aterms.id.values.astype(np.int)

    for sid, term, coll in aterms.values:
        try:
            if labeldf.loc[sid][term] != coll:
                labeldf.loc[sid][term] = coll
            labeldf.loc[sid][prev_term(term)] = 'I'
        except ValueError:  # some term was unknown in admissions cohort
            pass


def add_graduation_labels(labeldf, fname='nsf_degrees.csv'):
    deg = pd.read_csv(fname)
    gterms = deg.drop_duplicates('id')[['id', 'GRADTERM', 'degcoll']]
    gterms['id'] = gterms['id'].values.astype(np.int)

    # The label G indicates graduated
    for sid, term, coll in gterms.values:
        if labeldf.loc[sid][term] == 'N':
            labeldf.loc[sid][term] = 'G'
        else:
            labeldf.loc[sid][next_term(term)] = 'G'


def process_record(sid, record, labeldf):
    """Scan through and annotate gaps after enrollment as either
    NS (no summer) or B (break).
    """
    enrolled = False
    num_breaks = 0
    for term, label in record.iteritems():
        if label == 'G':
            break
        if num_breaks == 2:
            labeldf.loc[sid][term] = 'D'
            break

        if not enrolled:
            if label == 'I':
                enrolled = True
        else:
            if label == 'N':
                if str(term).endswith('40'):
                    labeldf.loc[sid][term] = 'NS'
                else:
                    labeldf.loc[sid][term] = 'B'
                    num_breaks += 1
            else:  # label is one of the colleges
                num_breaks = 0


def process_student(label_vector):
    """Read through `slabels` and return edges to add to the graph that
    represent the changing status of the student. `slabels` is a vector of
    labels, one for each term. The labels/nodes are 'I', 'G', or one of the 9
    colleges. Note that this does not include 'NS', 'N', or 'B' as labels of
    interest. We might at some point be interested in how often students from
    each college take breaks.
    """
    try:
        start_index = label_vector.tolist().index('I')
    except ValueError:
        raise StopIteration()

    cur_label = label_vector.ix[start_index]
    idx = start_index
    for label in label_vector[start_index:-1]:
        if label == 'G' or label == 'D':
            break
        if label == 'N':
            continue
        if label == 'NS' or label == 'B':
            label = cur_label

        next_label = label_vector.ix[idx+1]
        if next_label != 'NS' and next_label != 'B' and label != next_label:
            yield (label, next_label)
            cur_label = next_label

        idx += 1


def yield_edges(student_label_df):
    """Iterate over each row of the student labels data frame, yielding an edge
    for each college transition.
    """
    for sid, slabel_vector in student_label_df.iterrows():
        try:
            edges = process_student(slabel_vector)
            for edge in edges:
                yield edge
        except StopIteration:
            pass


def build_college_flow_graph(slabels):
    nodes = ('I','G','D', 'UN', 'LA', 'SC', 'BU', 'VS', 'AR', 'HH', 'E1', 'CA',
            'PP', 'EI', 'IC', 'LW')
    edges = yield_edges(slabels)
    g = igraph.Graph(directed=True)
    g.add_vertices(nodes)
    g.add_edges(edges)
    return g


def get_outweight(node):
    succs = [v.index for v in node.successors()]
    edge_pairs = zip([node.index]*len(succs), succs)
    edge_ids = [g.get_eid(*edge) for edge in edge_pairs]
    weights = [g.es[eid]['weight'] for eid in edge_ids]
    return sum(weights)


def get_inweight(node):
    preds = [v.index for v in node.predecessors()]
    edge_pairs = zip(preds, [node.index]*len(preds))
    edge_ids = [g.get_eid(*edge) for edge in edge_pairs]
    weights = [g.es[eid]['weight'] for eid in edge_ids]
    return sum(weights)


def prep_for_display(graph):
    graph.simplify(combine_edges=sum)
    graph.vs['inflow'] = map(get_inflow, graph.vs)
    graph.vs['outflow'] = map(get_outweight, graph.vs)
    return graph


def flow_matrix(graph):
    # combine multiple edges, so that the resulting edges
    # are weighted with the number of edges combined
    if not graph.is_weighted():
        graph.es['weight'] = 1
        g = graph.simplify(combine_edges=sum)
    else:
        g = graph

    # build the flow matrix, where cell (i, j) contains the
    # number of edges that moved from node i to node j (edge weight)
    num_nodes = len(graph.vs)
    matrix = np.ndarray((num_nodes, num_nodes))
    for v in graph.vs:
        for s in v.successors():
            eid = g.get_eid(v.index, s.index)
            matrix[v.index][s.index] = g.es[eid]['weight']

    # return the normalized matrix
    return matrix / matrix.sum()


def write_college_csv(fname='colleges.csv'):
    # TODO: add attributes to graph and write from graph
    with open(fname, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(('color', 'abbrev', 'name'))
        rows = [','.join(row) for row in COLLEGES]
        writer.writerows(rows)


def main():
    student_df = read_student_df()
    labeldf = label_students(student_df)

    # Next we need data from admissions and graduation
    add_admissions_labels(labeldf)
    add_graduation_labels(labeldf)

    # Next, scan through and annotate gaps after enrollment as either
    # NS (no summer) or B (break)
    for sid, record in labeldf.iterrows():
        process_record(sid, record, labeldf)

    # save the labels
    labeldf.to_csv('student-college-labels.csv')

    # Get graph edges
    graph = build_college_flow_graph(labeldf)
    graph.write_picklez('college-flow-graph.picklez')


if __name__ == "__main__":
    sys.exit(main())
