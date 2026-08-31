from rag import create_vector_store, search_policy


#vector_store = create_vector_store()

#results = vector_store.similarity_search("What is the maximum allowed portfolio VaR?", k=3)
results = search_policy('What is the maximum allowed portfolio VaR?')

for document in results:
    # print("PAGE:", document.metadata.get("page"))
    # print(document.page_content)
    print(document.page_content)
    print(document.metadata)
    print("-" * 80)
