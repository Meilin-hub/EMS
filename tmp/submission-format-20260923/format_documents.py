from pathlib import Path
from zipfile import ZipFile
from lxml import etree
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_LINE_SPACING

ROOT=Path.cwd()
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS={'w':W}
def tag(t): return '{'+W+'}'+t

def spacing(ppr):
    s=ppr.find(tag('spacing'))
    if s is None: s=etree.SubElement(ppr,tag('spacing'))
    s.set(tag('line'),'480');s.set(tag('lineRule'),'auto')
    for name,value in [('snapToGrid','0')]:
        el=ppr.find(tag(name))
        if el is None: el=etree.SubElement(ppr,tag(name))
        el.set(tag('val'),value)

def numbering(sect):
    for old in sect.findall(tag('lnNumType')): sect.remove(old)
    ln=etree.Element(tag('lnNumType'))
    for k,v in [('countBy','1'),('start','0'),('distance','120'),('restart','continuous')]: ln.set(tag(k),v)
    # ECMA-376 CT_SectPr order: lnNumType precedes pgNumType/cols/docGrid.
    following={'pgNumType','cols','formProt','vAlign','noEndnote','titlePg','textDirection','bidi','rtlGutter','docGrid','printerSettings','sectPrChange'}
    index=next((i for i,c in enumerate(sect) if etree.QName(c).localname in following),len(sect))
    sect.insert(index,ln)
    for grid in sect.findall(tag('docGrid')): grid.set(tag('type'),'default')

def patch_file(path):
    with ZipFile(path) as z: entries=[(i,z.read(i.filename)) for i in z.infolist()]
    output=[]
    for info,data in entries:
        is_header_footer=info.filename.startswith(('word/header','word/footer')) and info.filename.endswith('.xml')
        if info.filename not in ('word/document.xml','word/styles.xml') and not is_header_footer:
            output.append((info,data));continue
        tree=etree.fromstring(data)
        before=[x.text for x in tree.findall('.//w:t',NS)]
        if info.filename=='word/document.xml':
            for p in tree.findall('.//w:body//w:p',NS):
                pp=p.find(tag('pPr'))
                if pp is None: pp=etree.Element(tag('pPr'));p.insert(0,pp)
                spacing(pp)
                suppress=pp.find(tag('suppressLineNumbers'))
                if suppress is None: suppress=etree.SubElement(pp,tag('suppressLineNumbers'))
                has_text=bool(p.findall('.//w:t',NS))
                suppress.set(tag('val'),'0' if has_text else '1')
            for sec in tree.findall('.//w:sectPr',NS): numbering(sec)
        elif is_header_footer:
            for p in tree.findall('.//w:p',NS):
                pp=p.find(tag('pPr'))
                if pp is None: pp=etree.Element(tag('pPr'));p.insert(0,pp)
                suppress=pp.find(tag('suppressLineNumbers'))
                if suppress is None: suppress=etree.SubElement(pp,tag('suppressLineNumbers'))
                suppress.set(tag('val'),'1')
        else:
            for pp in tree.findall('.//w:pPr',NS):
                spacing(pp)
                for old in pp.findall(tag('suppressLineNumbers')): pp.remove(old)
        assert before==[x.text for x in tree.findall('.//w:t',NS)]
        output.append((info,etree.tostring(tree,xml_declaration=True,encoding='UTF-8',standalone=True)))
    temp=path.with_suffix('.tmp.docx')
    with ZipFile(temp,'w') as z:
        for info,data in output:z.writestr(info,data)
    temp.replace(path)
    print('Formatted',path.name)

for file in ['Journal_of_Hydrology_Abstract.docx','Journal_of_Hydrology_Cover_Letter.docx','Journal_of_Hydrology_Highlights.docx']:
    patch_file(ROOT/file)

doc=Document()
sec=doc.sections[0]
sec.page_width=Inches(8.27);sec.page_height=Inches(11.69)
sec.top_margin=sec.bottom_margin=Inches(1)
sec.left_margin=sec.right_margin=Inches(1)
for name in ['Normal','Title','Heading 1']:
    st=doc.styles[name]
    st.font.name='Times New Roman';st.font.size=Pt(12);st.font.color.rgb=RGBColor(0,0,0)
    st.paragraph_format.line_spacing_rule=WD_LINE_SPACING.DOUBLE
    st.paragraph_format.space_after=Pt(10)
doc.styles['Title'].font.size=Pt(14);doc.styles['Title'].font.bold=True
for st in doc.styles:
    for border in st.element.findall('.//'+qn('w:pBdr')): border.getparent().remove(border)
    if st.style_id in ['Title','TitleChar','Normal']:
        for fonts in st.element.findall('.//'+qn('w:rFonts')):
            for attr in list(fonts.attrib):
                if 'theme' in attr.lower(): del fonts.attrib[attr]
            for attr in ['ascii','hAnsi','eastAsia','cs']:fonts.set(qn('w:'+attr),'Times New Roman')
p=doc.add_paragraph('Declaration of Competing Interest','Title')
p.paragraph_format.keep_with_next=True
p=doc.add_paragraph();p.add_run('Manuscript title: ').bold=True
p.add_run('Graph Reinforcement Learning for Water-Energy-Food System under Observation Delays')
p=doc.add_paragraph();p.add_run('Authors: ').bold=True
p.add_run('Meilin Lyu, Andrew Huey Ping Tan, Shu-Hsiang Chen, and Wahbi El-Bouri')
p=doc.add_paragraph('The authors declare no competing financial interests or personal relationships that could have influenced, or appeared to influence, the work reported in this manuscript.')
doc.core_properties.title='Declaration of Competing Interest'
doc.core_properties.subject='Graph Reinforcement Learning for Water-Energy-Food System under Observation Delays'
doc.core_properties.author='Meilin Lyu; Andrew Huey Ping Tan; Shu-Hsiang Chen; Wahbi El-Bouri'
out=ROOT/'Declaration_of_Competing_Interest.docx'
doc.save(out);patch_file(out)
