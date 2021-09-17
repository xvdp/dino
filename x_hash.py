"""
hash
    binary content
    filename + filesize + savetime
    folder
"""
from typing import Any,Union, Collection
import os
import os.path as osp
import json
import hashlib
import torch


def hash_folder(files: Union[str,list], metadata: Collection=None,
                metakey: str="metadata", save: str=None, update: dict=None) -> dict:
    """
        Args
            files       folder name or file list
            [metadata]  iterable same len as files
            [metakey]   key indexing metadata
            [save]      filename to torch.save()
            [update]    other dict to merge with hashed folder
    Examples
    >>> features = hash_folder(files, m.features, "features", "mydir_features.pt")
    >>> features = torch.load("mydir_features.pt")
    """
    if isinstance(files, str) and osp.isdir(files):
        files = [f.path for f in os.scandir(files) if f.is_file()]
    if metadata is not None:
        assert len(metadata) == len(files)

    out = {}
    _meta = None
    for i, name in enumerate(files):
        if metadata is not None:
            _meta = metadata[i]
        out[name] = hash_file(name, splitname=True, metadata=_meta, metakey=metakey)

    if update is not None:
        out.update(update)

    if isinstance(save, str):
        save = save if osp.splitext(save)[-1] else save + ".pt"
        torch.save(out, save)
    return out

def hash_file(filename: str, splitname: bool=False,
              metadata: Any=None, metakey: str="metadata") -> dict:
    """
    hash dictionary
        name:       filename [if splitname: fullname]
        [folder]:   basedir if splitname
        datesize    md5 hash of tuple(mtime, size)
        content     md5 hash of file content
        <metakey>   any extra input

    """
    out = {}
    _st = os.stat(filename)
    if not splitname:
        out['name'] = filename
    else:
        out['folder'], out['name'] = osp.split(filename)
    out['datesize'] = hashlib.md5(json.dumps((_st.st_mtime,_st.st_size), sort_keys=True
                                  ).encode('utf-8')).hexdigest()
    with open(filename, 'rb') as _fi:
        out['content'] = hashlib.md5(_fi.read()).hexdigest()

    if metadata is not None:
        out[metakey] = metadata
    return out

def check_file(filename, hashdic):
    pass

def reversedict(dic: str, subkey:str, sort: bool=False) -> list:
    """
    given {key_0:{<subkey>:keyval0_i, ..., subkey_n:keyval0_n},
           ...,
           key_m:{<subkey>:keyvalm_i}, ....m}

    returns
        [(keyval0_i, key_0), ..., (keyvalm_1, key_m)]

    works on dict on which each value is a dict.
    assumes every item has same keys
    Exmaple
    >>> x = reversedict(dic, "content")

    """
    _subkeys = list(dic.keys())[0]
    assert subkey in dic[_subkeys].keys(), f"subkey '{subkey}' not found in {_subkeys}"

    out = [(dic[key][subkey], key) for key in dic]
    if sort:
        out = sorted(out, key=lambda x: x[0])
    return out

def get_keys_with_subkeyval(dic: dict, subkey: str, subval: str) -> dict:
    """ return dic subset with subkey:subval
    Example
    >> dic = torch.load("mydir.pt")
    >> get_keys_with_subkeyval(my_dic, "datesize", "c828d7d9d3aafd0b70127aae84208d97")
    """
    return {key:dic[key] for key in dic if dic[key][subkey] == subval}
