from dataclasses import dataclass, field
import pathlib
import pickle
import xml.etree.ElementTree as ET

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


@dataclass
class Document:
    doi: str = None
    sentences: list = field(default_factory=list)

    def save(self):
        if self.doi is None:
            raise RuntimeError("Cannot save file as it doesn't have a DOI number.")

        fname = pathlib.Path(f'{self.doi}.pkl')
        fname.parent.mkdir(parents=True, exist_ok=True)
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(doi):
        fname = pathlib.Path(f'{doi}.pkl')
        with open(fname, 'rb') as f:
            doc = pickle.load(f)
        return doc

    def __repr__(self):
        return f'<Document {self.doi} with {len(self.sentences)} sentences>'

    def as_array(self):
        return np.vstack([s.embedding for s in self.sentences])

    def __len__(self):
        return len(self.sentences)


@dataclass
class Sentence:
    text: str = None
    references: list = field(default_factory=list)
    garbled: bool = False
    embedding: list = None

    @property
    def trust(self):
        return len(self.references)

    def __str__(self):
        return self.text

    def __add__(self, other):
        s = Sentence(self.text + ' ' + other.text)
        s.references.extend(self.references)
        s.references.extend(other.references)
        s.garbled = self.garbled or other.garbled
        return s


def parse_document(filename):
    doi = None
    sentences = []
    bibliography = dict()

    parser = ET.XMLPullParser(['start', 'end'])
    parser.feed(open(filename).read())
    for event, elem in parser.read_events():
        if elem.tag == '{http://prismstandard.org/namespaces/basic/2.0/}doi' and event == 'start':
            doi = elem.text.replace(' ', '')
        elif elem.tag.endswith('para') and event == 'start':
            sentences.extend(parse_paragraph(parser, elem))
        elif elem.tag.endswith('bib-reference') and event == 'start':
            ref = parse_bib_reference(parser, elem)
            bibliography[ref['id']] = ref
        elif elem.tag.endswith('caption') and event == 'start':
            parse_caption(parser, elem)

    for sentence in sentences:
        references = []
        for ref in sentence.references:
            if ref in bibliography:
                references.append(bibliography[ref])
        sentence.references = references

    embeddings = get_embeddings(sentences)
    for s, emb in zip(sentences, embeddings):
        s.embedding = emb

    return Document(doi, sentences)


def parse_paragraph(parser, elem):
    in_progress = Sentence('')
    sentences, in_progress = to_sentences(in_progress, elem.text)
    for event, elem in parser.read_events():
        if elem.tag.endswith('para') and event == 'end':
            if elem.tail:
                finished, in_progress = to_sentences(in_progress, elem.tail)
                sentences.extend(finished)
            break
        elif (elem.tag.endswith('cross-ref') or elem.tag.endswith('cross-refs')) and event == 'start':
            references = parse_cross_ref(parser, elem)
            in_progress.references.extend(references)
        elif event == 'start':
            # Unknown element
            in_progress.garbled = True
        elif event == 'end':
            if elem.tail:
                finished, in_progress = to_sentences(in_progress, elem.tail)
                sentences.extend(finished)
    return sentences


def parse_cross_ref(parser, elem):
    if 'refid' in elem.attrib:
        return elem.attrib['refid'].split()
    else:
        return []


def parse_bib_reference(parser, elem):
    ref = dict(id=elem.attrib['id'])
    for event, elem in parser.read_events():
        if elem.tag.endswith('bib-reference') and event == 'end':
            break
        elif elem.tag.endswith('doi') and event == 'start':
            ref['doi'] = elem.text.replace(' ', '')
        elif elem.tag.endswith('label') and event == 'start':
            ref['label'] = elem.text
    return ref


def parse_caption(parser, elem):
    for event, elem in parser.read_events():
        if elem.tag.endswith('caption') and event == 'end':
            break


def to_sentences(in_progress, text):
    text = text.strip()
    if text.startswith(')'):
        text = text[1:]
    if text.endswith('('):
        text = text[:-1]
    if text.endswith('(see'):
        text = text[:-4]
    [*text_finished, text_in_progress] = text.split('. ')
    finished_sentences = [Sentence(t) for t in text_finished]

    # Can we finish the sentence in progress?
    if len(text_finished) > 0:
        finished_sentences[0] = in_progress + finished_sentences[0]
        in_progress = Sentence(text_in_progress)
    else:
        in_progress += Sentence(' ' + text_in_progress)

    return finished_sentences, in_progress


def get_embeddings(sentences):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
    tokens = []
    attention_mask = []
    for sentence in sentences:
        new_tokens = tokenizer.encode_plus(sentence.text, max_length=128,
                                           truncation=True, padding='max_length',
                                           return_tensors='pt')
        tokens.append(new_tokens['input_ids'][0])
        attention_mask.append(new_tokens['attention_mask'][0])
    tokens = torch.stack(tokens)
    attention_mask = torch.stack(attention_mask)

    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
    with torch.no_grad():
        outputs = model(input_ids=tokens, attention_mask=attention_mask)
    word_embeddings = outputs.last_hidden_state.numpy()
    sentence_embeddings = np.mean(attention_mask.numpy()[:, :, None] * word_embeddings, axis=1)
    return sentence_embeddings
