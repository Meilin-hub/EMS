from pathlib import Path
import re,json
from collections import Counter
old=Path('tmp/layout12/before/original.tex').read_text()
new=Path('cas-dc-sample.tex').read_text()
def clean(s):return re.sub(r'(?<!\\)%[^\n]*','',s)
def normalized(s):return re.sub(r'\s+','',s)
def main(s):
 s=clean(s).split('\\begin{document}',1)[1].split('\\appendix',1)[0]
 return normalized(s.replace('\\flushcolsend',''))
assert main(old)==main(new),'Main text/equations/figures/tables changed'
def args(s,cmd):
 s=clean(s);out=[]
 for m in re.finditer(r'\\'+cmd+r'\{',s):
  start=i=m.end();depth=1
  while depth:
   if s[i]=='{':depth+=1
   elif s[i]=='}':depth-=1
   i+=1
  out.append(s[start:i-1])
 return out
assert Counter(args(old,'caption'))==Counter(args(new,'caption')),'Caption wording changed'
assert Counter(args(old,'label'))==Counter(args(new,'label')),'Label missing'
assert Counter(args(old,'citep'))==Counter(args(new,'citep')),'Citation changed'
for label in ['tab:definitions','tab:system-performance-indicators']:
 def body(s):
  b=clean(s).split('\\label{'+label+'}',1)[1]
  return b.split('\\toprule',1)[1].split('\\bottomrule',1)[0]
 assert body(old)==body(new),label+' cells changed'
start='Water & Reservoir storage and loss:'
s3old=clean(old).split(start,1)[1].split('\\bottomrule',1)[0]
s3new=clean(new).split(start,1)[1].split('\\end{xltabular}',1)[0]
assert s3old==s3new,'Source table cells changed'
assert Path('tmp/layout12/before/original.bbl').read_bytes()==Path('cas-dc-sample.bbl').read_bytes(),'Bibliography changed'
figs=re.findall(r'\\begin\{figure\}(.*?)\\end\{figure\}',clean(new),re.S)
assert len(figs)==6 and all('\\columnwidth' in x for x in figs)
mainpart=new.split('\\appendix')[0]
assert len(re.findall(r'\\begin\{table\}',mainpart))==3 and '\\begin{table*}' not in mainpart
report={'main_text_equations_figures_tables_unchanged':True,'captions_unchanged':True,'supplementary_table_cells_unchanged':True,'bibliography_unchanged':True,'labels_and_citations_unchanged':True,'figures_single_column':6,'main_tables_single_column':3}
Path('tmp/layout12/content-check.json').write_text(json.dumps(report,indent=2))
print(report)
