import sbol2
import os
from rdflib import Graph
import pandas as pd
import numpy as np
from rdflib.query import ResultRow
import sys 
import configparser

def get_sequence_from_sbol(file_path):
    """
    Extracts the nucleotide sequence from an SBOL XML file.

    The function parses an SBOL file into an RDF graph and runs a SPARQL query to find
    the `sbol:elements` property, which stores the sequence string (e.g., DNA bases).
    It returns the FIRST sequence found as a string, or `None` if no sequence is present.

    Parameters
    ----------
    file_path : str
        Path to the SBOL XML file to parse.

    Returns
    -------
    str or None
        The extracted sequence string if found, otherwise None.
    """
     
    g = Graph()
    g.parse(file_path, format="xml")

    sparql_query ='''PREFIX sbol: <http://sbols.org/v2#>

    SELECT ?sequence
    WHERE {
    ?s sbol:elements ?sequence .
    }
    '''
    query_result = g.query(sparql_query)

    if query_result:
        for row in query_result:
            if isinstance(row, ResultRow):
                return str(row.sequence)
            
    print("No sequence found.")
    return None

def find_possible_y_uris(file_path):
    """
    Identifies candidate predicate URIs associated with numeric values (for the y labels) in an SBOL XML file.

    Parameters
    ----------
    file_path : str
        Path to the SBOL XML file to parse.

    Returns
    -------
    list of str
        A list of predicate URIs that are linked to numeric values.
    """
    g = Graph()
    g.parse(file_path, format="xml")

    sparql_query ='''
    SELECT DISTINCT ?hasValue ?value
    WHERE {
    ?item ?hasValue ?value .
    }
    '''
    query_result = g.query(sparql_query)

    possible_y_uris = []
    if query_result:
        for row in query_result:
            if isinstance(row, ResultRow):
                try:
                    res = float(row.value)
                    possible_y_uris.append(str(row.hasValue))
                except:
                    continue

    else:
        print("Query failed.")

    # TODO allow the user to see non-numeric y labels as well

    return possible_y_uris





def get_y_label(file_path, uri="http://www.ontology-of-units-of-measure.org/resource/om-2/hasNumericalValue"):
    """
    Extracts a numerical label from an SBOL XML file using a specified predicate URI.

    Parameters
    ----------
    file_path : str
        Path to the SBOL XML file to parse.
    uri : str, optional
        The predicate URI used to locate the numerical value.
        Defaults to "http://www.ontology-of-units-of-measure.org/resource/om-2/hasNumericalValue".

    Returns
    -------
    float or None
        The first numerical value found as a float, or None if no value is found.
    """

    g = Graph()
    g.parse(file_path, format="xml")

    sparql_query = f'''

    SELECT ?numericalValue
    WHERE {{
    ?s <{uri}> ?numericalValue .
    }}
    '''
    query_result = g.query(sparql_query)

    if query_result:
        for row in query_result:
            if isinstance(row, ResultRow):
                return float(row.numericalValue) 
    
    print("No numerical values found.")
    return None

def get_sequences_from_sbol(file_paths):
    """
    Extracts sequences from multiple SBOL XML files.

    Parameters
    ----------
    file_paths : list of str
        A list of paths to SBOL XML files to parse.

    Returns
    -------
    list of str
        A list containing the extracted sequence from each file.

    Raises
    ------
    ValueError
        If any file does not contain a sequence.
    """
    all_sequences = []
    for file_path in file_paths:
        sequence = get_sequence_from_sbol(file_path)
        if sequence is None:
            raise ValueError(f"No sequence found in {file_path}.")

        all_sequences.append(sequence)
    return all_sequences

def get_y_labels_from_sbol(file_paths, uri):
    """
    Extracts numerical labels from multiple SBOL XML files using a specified predicate URI.

    Parameters
    ----------
    file_paths : list of str
        A list of paths to SBOL XML files to parse.
    uri : str
        The predicate URI used to locate the numerical values.

    Returns
    -------
    list of float
        A list of numerical labels extracted from each file.

    Raises
    ------
    ValueError
        If any file does not contain a numerical label for the given URI.
    """
    y_labels = []
    for file_path in file_paths:
        y_label = get_y_label(file_path, uri)
        if y_label is None:
            raise ValueError(f"No y label found in {file_path}.")
        y_labels.append(y_label)
    return y_labels

def build_dataset(file_paths, y_uri):
    """
    Builds a dataset from SBOL XML files by pairing sequences with numerical labels.

    Parameters
    ----------
    file_paths : list of str
        A list of paths to SBOL XML files to parse.
    y_uri : str
        The predicate URI used to extract numerical labels.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with two columns:
        - "sequence": the extracted sequence string.
        - "target": the corresponding numerical label.
    """
    y_labels = get_y_labels_from_sbol(file_paths, y_uri)
    sequences = get_sequences_from_sbol(file_paths)
    df = pd.DataFrame({"sequence": sequences, "target": y_labels})
    return df