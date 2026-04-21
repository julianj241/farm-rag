from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

vs = Chroma(
    collection_name='field_log',
    persist_directory='./chroma_db',
    embedding_function=OllamaEmbeddings(model='nomic-embed-long'),
)

results = vs.similarity_search('yellowing leaves on JJ beds overwatered', k=3)
for d in results:
    month = d.metadata.get('month')
    year = d.metadata.get('year')
    part = d.metadata.get('part', 1)
    snippet = d.page_content[:200].replace('\n', ' ')
    print(f'{month} {year} (part {part}): {snippet}...')
    print('---')