import sys
import textwrap

from scipy.spatial import distance
import numpy as np
import colorama

from document import Document


colorama.init()

doc1 = Document.load(sys.argv[1])
doc2 = Document.load(sys.argv[2])
doc2_embeddings = doc2.as_array()

for i, s in enumerate(doc1.sentences):
    if any([r.get('doi', '') == doc2.doi for r in s.references]):
        distances = distance.cdist([s.embedding], doc2_embeddings, metric='cosine')[0]
        ranking = np.argsort(distances)

        for line in textwrap.wrap(f'{i:03d}: {s}', width=80, subsequent_indent='     '):
            print(colorama.Fore.GREEN + line + colorama.Style.RESET_ALL)
        print(f"From {s.references[0]['label']}:")
        for rank, j in enumerate(ranking[:5]):
            for line in textwrap.wrap(f'  {rank} {distances[j]:.3f} {doc2.sentences[j]}',
                                      width=80, subsequent_indent='          '):
                print(line)
        print('\n')
