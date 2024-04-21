'''collection of scripts for easily and quickly grabbing info from ABA structure tree

 ／l、
（ﾟ､ ｡ ７
  l  ~ヽ
  じしf_,)ノ
'''

def get_name_for_id(id):
    '''returns ABA name for region ID'''
    from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
    mcc = MouseConnectivityCache()
    structure_tree = mcc.get_structure_tree()
    return structure_tree.get_structures_by_id([reg])[0]['name']

def generate_ABA_df():
    from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
    import pandas as pd

    mcc = MouseConnectivityCache()
    structure_tree = mcc.get_structure_tree()

    s = structure_tree.get_structure_sets()
    sets = structure_tree.get_structures_by_set_id([i for i in s])
    return pd.DataFrame(sets)

def search_for_name(str):
    df = generate_ABA_df()
    return df[df['name'].str.lower().str.contains(str)]
