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
            text += elem.text.strip()
            break;
        elif elem.tag.endswith('cross-ref') and event == 'start':
            references.append(parse_cross_ref(parser, elem))
        elif elem.text:
            text += elem.text.strip()
    return text, references

def parse_cross_ref(parser, elem):
    return elem.attrib

paragraphs = parse_document('./paper.xml')
