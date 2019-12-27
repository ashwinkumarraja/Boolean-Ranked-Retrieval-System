Boolean Retrieval System

This is an implementation of a simple boolean retrieval system.

The query is taken from the user , preprocessed - which involves tokenizing it , removing punctuations and casing etc.

The whole corpus is intitally processed and the tf-idf matrix is formed. The rank of the document is decided with respect to the query with the initial tf-idf weight and a proximity weight which gives a better ranked list of results.

The output is a set of documents in which the query is present in the decreasing order of the weight.
