from pathlib import Path
p=Path('cas-dc-sample.tex')
s=p.read_text()
s=s.replace('\\usepackage{flushend}', '\\usepackage{flushend}\n\\usepackage{xltabular}\n\\usepackage{multicol}')
start=s.index('\\flushcolsend\n\\appendix')
end=s.index('\\bibliographystyle{cas-model2-names}',start)
app=s[start:end]
a=app.index('\\begin{table*}[pos=t]')
b=app.index('\\end{table*}',a)+len('\\end{table*}')
s1=app[a:b]
a2=app.index('\\noindent\\begin{minipage}{\\columnwidth}')
b2=app.index('\\end{minipage}',a2)+len('\\end{minipage}')
s2=app[a2:b2]
a3=app.index('\\begin{table*}[pos=t]',b2)
b3=app.index('\\end{table*}',a3)+len('\\end{table*}')
s3=app[a3:b3]
section='\\section{Supplementary tables}\n\\label{app:supplementary-tables}'
intro='This appendix contains the system performance indicators, and the data source details referenced in the main text.'
caption_setup=r'''\makeatletter\def\@captype{table}\makeatother
\ExplSyntaxOn
\__reset_tbl:
\dim_set:Nn \l_tbl_width_dim { \linewidth }
\cs_set_eq:cN { @makecaption } \__make_tbl_caption:nn
\ExplSyntaxOff'''
s2=s2.replace('\\noindent\\begin{minipage}{\\columnwidth}', '\\noindent\\begin{minipage}{\\columnwidth}\n'+section+'\n\n'+intro+'\n\\par\\smallskip')
s1=s1.replace('\\begin{table*}[pos=t]\n{\\normalfont\\normalsize\n'+section+'}', '\\noindent\\begin{minipage}{\\textwidth}\n'+caption_setup)
s1=s1.replace('\\end{table*}', '\\end{minipage}')
s1=s1.replace('\\centering\n\\footnotesize', '\\centering\n\\sffamily\\footnotesize')
# A page-breakable data-source table retains all original cells.
caption=s3[s3.index('\\caption'):s3.index('\\centering')].strip()
tabstart=s3.index('\\begin{tabularx}')
headend=s3.index('\\toprule')
preamble=s3[tabstart:headend].replace('tabularx','xltabular')
body=s3[headend:s3.index('\\end{tabularx}')]
header='\\toprule\nData category & Variables & Source \\\\\n\\midrule\n'
assert body.startswith(header)
body=body[len(header):]
body=body.rsplit('\\bottomrule',1)[0]
longtable=r'''\begingroup
\sffamily\footnotesize
\renewcommand{\arraystretch}{1.00}
\setlength{\LTpre}{8pt}
\setlength{\LTpost}{6pt}
\setlength{\LTcapwidth}{\textwidth}
\makeatletter
\long\def\LT@makecaption#1#2#3{%
  \LT@mcol\LT@cols c{\hbox to\z@{\hss\parbox[t]\LTcapwidth{%
    \sffamily\small\raggedright
    #1{\textbf{#2}\par}#3\par\vskip4pt}\hss}}}
\makeatother
'''+preamble+caption+' \\\\\n'+header+r'''\endfirsthead
\multicolumn{3}{@{}l}{\sffamily\small\textbf{\tablename~\thetable}} \\
'''+header+r'''\endhead
\midrule
\endfoot
\bottomrule
\endlastfoot
'''+body+r'''\end{xltabular}
\endgroup
'''
# Keep the unused commented audit template unchanged.
comments=app[b2:a3]
comments=comments.replace('\\par\\medskip\n','',1)
newapp=r'''\appendix
\setcounter{table}{0}
\renewcommand{\thetable}{S\arabic{table}}

% Keep the short performance-indicator table with the appendix introduction.
'''+s2+r'''
\par
\flushcolsend
\onecolumn

% Full-width supplementary tables share the page flow; the long table may continue.
'''+s1+'\n'+comments+'\n'+longtable+'\n'+r'''% Continue references immediately in two columns, without a forced page break.
\begin{multicols}{2}
'''
tail=s[end:]
tail=tail.replace('\\bibliography{references}\n\\flushcolsend','\\bibliography{references}\n\\end{multicols}')
s=s[:start]+newapp+tail
p.write_text(s)
