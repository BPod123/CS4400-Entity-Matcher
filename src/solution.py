from os.path import abspath, join
from bisect import bisect_left
from src.data_handler import *
from itertools import product



def find_matches(tests, ltable, rtable, rf):
    classifications = rf.predict([get_data_from_ids(t[0], t[1], ltable, rtable) for t in tests])
    return [tests[i] for i in range(len(tests)) if classifications[i]]




def block_by_attr(ltable: pd.DataFrame, rtable: pd.DataFrame, attr="brand", secondary_search_in="title"):
    """

    :param ltable: Dataframe
    :param rtable: Dataframe
    :param attr: The key of the tables to do the blocking by
    :param secondary_search_in: If the attribute is null, search in a secondary attribute for the value
    :return: A list of tuples of the form (l_id_table, r_id_table) containing indices in the left and right tables
            of items with the same brand.
            For items without a brand, any brand names that show up in their title will cause them to be placed in
            the that brand's table.
    """
    null_attrs = {"", "- na -", "-na-", "nan", None}
    l_attrs = set(ltable[attr].astype(str).values)
    r_attrs = set(rtable[attr].astype(str).values)
    attrs = l_attrs.union(r_attrs)
    l_attr_ids = {b.lower(): set() for b in l_attrs if b not in null_attrs}
    r_attr_ids = {b.lower(): set() for b in r_attrs if b not in null_attrs}
    l_missing_attrs = []
    r_missing_attrs = []

    for table, attrs_ids, missing_attrs in [
        (ltable, l_attr_ids, l_missing_attrs), (rtable, r_attr_ids, r_missing_attrs)]:
        for i, x in table.iterrows():
            val = str(x[attr]).lower()
            if val in null_attrs:
                missing_attrs.append(i)
            else:
                attrs_ids[val].add(i)
    for table, attr_ids, missing_attr in [(ltable, l_attr_ids, l_missing_attrs), (rtable, r_attr_ids, r_missing_attrs)]:
        for item_id in missing_attr:
            second = str(table[secondary_search_in][item_id]).lower()
            second_split = second.split(" ")
            attr_indices = []
            for key in attrs_ids.keys():
                key_tokens = key.split(" ")
                min_index = float("inf")
                for token in key_tokens:
                    if token in second_split:
                        min_index = min(min_index, second_split.index(token))
                if min_index != float("inf"):
                    attr_indices.append((key, min_index))
            attr_indices.sort(key=lambda x: x[1])
            if len(attr_indices) > 0:
                if attr_indices[0][0] not in attr_ids.keys():
                    attr_ids[attr_indices[0][0]] = {item_id}
                else:
                    attr_ids[attr_indices[0][0]].add(item_id)
    # At this point every attribute in every table has been assigned a brand
    # However, there are brands in ltable not in rtable and vice versa
    # Not that every item has been assigned by attribute, need to work out differences between table attrs
    shared_attrs = set(l_attr_ids.keys()).intersection(r_attr_ids.keys())
    for table_1, attr_ids_1, table_2, attr_ids_2 in [(ltable, l_attr_ids, rtable, r_attr_ids),
                                                     (rtable, r_attr_ids, ltable, l_attr_ids)]:
        # set(attr_ids_2.keys()).difference(set(attr_ids_1.keys()))
        missing_attr_keys = set(attr_ids_1.keys()).difference(set(attr_ids_2.keys()))

        similar_attrs = [(m, {x for x in shared_attrs if len({x.split(" ")[0]}
                                                             .intersection({m.split(" ")[0]})) > 0 or (
            m in x.split(" "))}) for m in missing_attr_keys]
        similar_attrs = [x for x in similar_attrs if len(x[1]) > 0]

        for missing_attr_key, similar_set in similar_attrs:
            value_set = attr_ids_1.pop(missing_attr_key)
            for similar_attr in similar_set:
                if similar_attr not in attr_ids_1.keys():
                    continue
                elif attr_ids_1[similar_attr] is None:
                    attr_ids_1[similar_attr] = value_set
                else:
                    attr_ids_1[similar_attr].update(value_set)

    left_keys_only = [x for x in set(l_attr_ids.keys()).difference(set(r_attr_ids.keys()))]
    right_keys_only = [x for x in set(r_attr_ids.keys()).difference(set(l_attr_ids.keys()))]
    left_keys_only.sort()
    right_keys_only.sort()
    # Now to combine attributes (x,y) if they are in the form (name, name inc) or (name, name corporation), etc...
    common_suffixes = "corporation incorporated company corp inc. co. ltd. limited products technologies tech".split(" ")
    for attr_id_dict, only_keys, other_ids_dict in [(l_attr_ids, left_keys_only, r_attr_ids), (r_attr_ids, right_keys_only, l_attr_ids)]:
        side_shared_keys = set(attr_id_dict.keys()).difference(only_keys)
        for suffix in common_suffixes:
            for key in {x for x in only_keys if " {0} ".format(x).endswith(" {0} ".format(suffix))}:
                key_minus_suffix = key.replace(" {0}".format(suffix), "")
                if key_minus_suffix in side_shared_keys.union(other_ids_dict.keys()):
                    value_set = attr_id_dict.pop(key)
                    if key_minus_suffix in attr_id_dict.keys():
                        attr_id_dict[key_minus_suffix].update(value_set)
                    else:
                        attr_id_dict[key_minus_suffix] = value_set

    for key in {x for x in set(l_attr_ids.keys()).union(set(r_attr_ids.keys())) if x not in shared_attrs}:
        if key in l_attr_ids.keys():
            l_attr_ids.pop(key)
        else:
            r_attr_ids.pop(key)
    prods = (product(l_attr_ids[key], r_attr_ids[key]) for key in shared_attrs)
    ret_val = []
    [ret_val.extend(prod) for prod in prods]
    return ret_val



def chunk_list(lst, num_chunks):
    divs = list()
    chunk_length = int(len(lst) / num_chunks)
    for i in range(0, len(lst), chunk_length):
        divs.append(lst[i: i + chunk_length])
    return divs

def run_solution(training_data: pd.DataFrame, ltable: pd.DataFrame, rtable: pd.DataFrame, class_name="label", output_file=abspath("../out/output.csv")):
    """

    :param training_data: Table in the form [[ltable_id], [rtable_id], [class_name]]
    :param ltable: pandas DataFrame where each row is equal to the id of the item in that row
    :param rtable: pandas DataFrame where each row is equal to the id of the item in that row
    :param class_name: Attribute key associated with the correct classification for each example in the training data
    :return: creates the output.csv file with the results from testing everything that is not training data
    """
    #*****************************
    training_tups = {(training_data["ltable_id"][i], training_data["rtable_id"][i]) for i in range(len(training_data))}

    examples, labels = generate_examples(training_data, ltable, rtable, class_name)



    # training_label = training_data[class_name]
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    # rf = RandomForestClassifier(class_weight="balanced", random_state=0)
    # rf = RandomForestClassifier()

    rf = RandomForestRegressor(min_samples_leaf=100)
    rf.fit(examples, labels)

    tests = [x for x in block_by_attr(ltable, rtable)if x not in training_tups]
    matching_ids = find_matches(tests, ltable, rtable, rf)
    outut_dict = {"ltable_id": [x[0] for x in matching_ids], "rtable_id": [x[1] for x in matching_ids]}
    output = pd.DataFrame(data=outut_dict)
    output.to_csv(output_file, index=False)




if __name__ == "__main__":
    ltable = pd.read_csv(join(abspath('../data'), "ltable.csv"))
    rtable = pd.read_csv(join(abspath('../data'), "rtable.csv"))
    training_data = pd.read_csv(join(abspath('../data'), "train.csv"))
    run_solution(training_data, ltable, rtable)
