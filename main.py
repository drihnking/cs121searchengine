# Dictionary to trace back document url given docid: key=docid, val=page_url
PAGES_LIST = {} 

def pages_index(pages):
    docid = 0
    for page_url in pages:
        docid += 1
        PAGES_LIST[page_url] = docid

def build_index(pages):
    '''
    Indexes pages containing each token t
    '''
    inverted_list = {} # dictionary with key=token, value = docid
    
    # Parse each text document in "documents" list
    for page_url in pages:
        # Parse the document into tokens
        tokens = parse(doc)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            inverted_list[token] = inverted_list.get(token, []).append(docid)
    
    return inverted_list
