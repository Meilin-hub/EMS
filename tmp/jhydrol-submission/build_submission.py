from pathlib import Path
from copy import deepcopy
from zipfile import ZipFile
from hashlib import sha256
from lxml import etree
import json

ROOT = Path('/Users/maylinlyu/Documents/PHD/p4/code1/latex1')
QA = ROOT / 'tmp/jhydrol-submission'
REF = Path('/Users/maylinlyu/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-legal-memorandum/assets/reference.docx')
OUT = ROOT / 'output/submission'
OUT.mkdir(parents=True, exist_ok=True)
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W}
def q(x): return '{%s}%s' % (W, x)
def element(tag, **attrs):
    e = etree.Element(q(tag))
    for k, v in attrs.items(): e.set(q(k), str(v))
    return e

title = 'Graph Reinforcement Learning for Water-Energy-Food System under Observation Delays'
paragraphs = [
    f'Please consider our manuscript, "{title}", for publication as a Research Paper in Journal of Hydrology.',
    'Delayed observations can compromise reservoir releases and irrigation allocation across interconnected water, energy and food systems. We address this problem in the Upper Yellow River by coupling a heterogeneous graph neural network with proximal policy optimization. The graph represents 24 connected entities, while a deterministic execution layer enforces reservoir, channel, diversion and pumping-energy constraints. Monthly data from 2001 to 2021 support training, validation and evaluation over separate periods.',
    'In the held-out evaluation, the graph policy achieved an approximately 6.2% lower average water-energy-food penalty than conventional proximal policy optimization across uniform observation delays of one to five months (p = 0.027). Performance without delays was statistically comparable. The graph policy maintained near-baseline performance under the tested delays, but single-node analysis identified the Hetao diversion interface as the most delay-sensitive location, indicating a priority for timely monitoring.',
    'The manuscript fits Journal of Hydrology through its focus on basin-scale water resources operation and the links among reservoir regulation, irrigation supply, pumping energy and downstream flow. It offers a framework for assessing how information delays affect coupled river systems and where monitoring improvements may most benefit operational robustness. These questions are relevant to other regulated basins with observations reported at different times.'
]
bullets = [
    'A 24-node graph links water, energy and food systems in the Upper Yellow River.',
    'Physical constraints keep reservoir releases and irrigation diversions feasible.',
    'Graph control cuts average penalties by 6.2% versus a standard policy under delays.',
    'System performance stays near baseline with observation delays up to five months.',
    'The Hetao diversion interface is most sensitive to delayed observations.'
]
assert len(bullets) == 5 and all(len(s) + 2 <= 85 for s in bullets)

with ZipFile(REF) as z:
    archive = {i.filename: (i, z.read(i.filename)) for i in z.infolist()}
source_root = etree.fromstring(archive['word/document.xml'][1])
source_ps = source_root.findall('w:body/w:p', NS)

def make_p(index, text, title_style=False, compact=False, list_id=None):
    p = deepcopy(source_ps[index])
    p.attrib.clear()
    ppr = p.find('w:pPr', NS)
    rp = p.find('w:r/w:rPr', NS)
    rpr = deepcopy(rp) if rp is not None else etree.Element(q('rPr'))
    if index == 3:
        for b in rpr.findall('w:b', NS) + rpr.findall('w:bCs', NS): rpr.remove(b)
    for ch in list(p):
        if ch is not ppr: p.remove(ch)
    if title_style:
        ppr.insert(0, element('pStyle', val='Title'))
    if compact:
        spacing = ppr.find('w:spacing', NS)
        if spacing is None: spacing = etree.SubElement(ppr, q('spacing'))
        spacing.set(q('after'), '0')
        keep = ppr.find('w:keepNext', NS)
        if keep is None: keep = etree.SubElement(ppr, q('keepNext'))
        keep.set(q('val'), '1')
    if list_id is not None:
        ppr.find('w:numPr/w:numId', NS).set(q('val'), str(list_id))
        ppr.find('w:keepLines', NS).set(q('val'), '1')
        ppr.find('w:spacing', NS).set(q('after'), '160')
        ppr.find('w:jc', NS).set(q('val'), 'left')
    r = etree.SubElement(p, q('r'))
    r.append(rpr)
    t = etree.SubElement(r, q('t'))
    t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    t.text = text
    return p

def serialize(e):
    return etree.tostring(e, encoding='UTF-8', xml_declaration=True, standalone=True)

def build(kind):
    root = deepcopy(source_root)
    body = root.find('w:body', NS)
    sect = deepcopy(body.find('w:sectPr', NS))
    for ch in list(body): body.remove(ch)
    changes = {}
    if kind == 'Cover_Letter':
        body.append(make_p(1, 'COVER LETTER', title_style=True))
        body.append(make_p(3, 'To: The Editors, Journal of Hydrology'))
        body.append(make_p(6, 'Date: 9 September 2026'))
        body.append(make_p(3, 'Dear Editors,'))
        for text in paragraphs: body.append(make_p(11, text))
        body.append(make_p(3, 'Thank you for considering our manuscript.'))
        body.append(make_p(3, 'Sincerely,', compact=True))
        for text in [
            'Andrew Huey Ping Tan',
            'Corresponding author',
            'School of Intelligent Manufacturing Ecosystem',
            "Xi'an Jiaotong-Liverpool University, Suzhou 215412, China",
            'Email: Andrew.Tan@xjtlu.edu.cn'
        ]:
            body.append(make_p(3, text, compact=True))
    else:
        numbering = etree.fromstring(archive['word/numbering.xml'][1])
        aid = max(int(n.get(q('abstractNumId'))) for n in numbering.findall('w:abstractNum', NS)) + 1
        nid = max(int(n.get(q('numId'))) for n in numbering.findall('w:num', NS)) + 1
        abstract = element('abstractNum', abstractNumId=aid)
        abstract.append(element('multiLevelType', val='singleLevel'))
        lvl = element('lvl', ilvl=0)
        lvl.append(element('start', val=1))
        lvl.append(element('numFmt', val='bullet'))
        lvl.append(element('lvlText', val='•'))
        lvl.append(element('lvlJc', val='left'))
        pp = element('pPr')
        pp.append(element('ind', left=1440, hanging=360))
        lvl.append(pp)
        rp = element('rPr')
        rp.append(element('rFonts', ascii='Garamond', hAnsi='Garamond'))
        lvl.append(rp)
        abstract.append(lvl)
        numbering.insert(0, abstract)
        num = element('num', numId=nid)
        num.append(element('abstractNumId', val=aid))
        numbering.append(num)
        changes['word/numbering.xml'] = serialize(numbering)
        body.append(make_p(1, 'HIGHLIGHTS', title_style=True))
        for text in bullets: body.append(make_p(14, text, list_id=nid))
    body.append(sect)
    changes['word/document.xml'] = serialize(root)
    header = etree.fromstring(archive['word/header1.xml'][1])
    for t in header.findall('.//w:t', NS): t.text = ''
    changes['word/header1.xml'] = serialize(header)
    settings = etree.fromstring(archive['word/settings.xml'][1])
    update = settings.find('w:updateFields', NS)
    if update is None: update = etree.SubElement(settings, q('updateFields'))
    update.set(q('val'), 'true')
    changes['word/settings.xml'] = serialize(settings)
    path = OUT / f'Journal_of_Hydrology_{kind}.docx'
    with ZipFile(path, 'w') as z:
        for name, (info, data) in archive.items(): z.writestr(info, changes.get(name, data))
    with ZipFile(path) as z:
        for name, (_, data) in archive.items():
            if name not in changes: assert z.read(name) == data, name
    print(path)
    return path

files = [build('Cover_Letter'), build('Highlights')]
(QA / 'content-and-character-check.json').write_text(json.dumps({
    'source': str(ROOT / 'cas-dc-sample.tex'),
    'manuscript_title': title,
    'cover_letter_paragraphs': paragraphs,
    'cover_letter_body_words': sum(len(s.split()) for s in paragraphs),
    'highlights': [{'text': s, 'characters_with_spaces': len(s), 'with_bullet_and_space': len(s)+2} for s in bullets],
    'files': [str(p) for p in files],
    'reference_sha256': sha256(REF.read_bytes()).hexdigest(),
    'claims': {'method': 'Introduction and sections 2-3', '6.2_percent': 'Abstract, Table 2 and Conclusions', 'held_out_period': 'Section 3.2 Training and evaluation procedure', 'delay_robustness': 'Experiments 2-3', 'Hetao_interface': 'Experiment 3, Fig. 5'},
    'journal_sources': ['https://www.elsevier.com/researcher/author/tools-and-resources/highlights', 'https://www.elsevier.support/publishing/answer/what-should-be-included-in-a-cover-letter', 'https://shop.elsevier.com/journals/journal-of-hydrology/0022-1694'],
    'guide_access': 'ScienceDirect guide blocked by human verification; publisher guidance and official scope verified.'
}, indent=2))
print('Body words:', sum(len(s.split()) for s in paragraphs))
print('Highlight character counts:', [len(s) for s in bullets])
