import xml.etree.ElementTree as ET


def parse_document(filename):
    parser = ET.XMLPullParser(['start', 'end'])
    parser.feed(open(filename).read())
    paragraphs = []
    for event, elem in parser.read_events():
        if elem.tag.endswith('para') and event == 'start':
            paragraphs.append(parse_paragraph(parser, elem))
        elif elem.tag.endswith('cross_ref') and event == 'start':
            reference = parse_cross_ref(parser, elem)
            paragraphs[-1][1].append(reference)
    return paragraphs

def parse_paragraph(parser, elem):
    text = elem.text.strip()
    references = []
    for event, elem in parser.read_events():
        if elem.tag.endswith('para') and event == 'end':
            if elem.tail:
                text += elem.tail.strip()
            break;
        elif elem.tag.endswith('cross-ref') and event == 'start':
            reference, tail = parse_cross_ref(parser, elem)
            references.append(reference)
            if tail:
                text += tail
        elif event == 'end':
            if elem.tail:
                text += elem.tail.strip()
    return text, references

def parse_cross_ref(parser, elem):
    return elem.attrib, elem.tail

#breakpoint()
paragraphs = parse_document('./snippet.xml')
