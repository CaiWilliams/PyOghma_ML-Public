import io
import os

class Document:
    def __init__(self, document_class, document_properties, packages, **kwargs):
        self.file = io.StringIO()
        self.tab = 0
        self.newline = True

        self.document_class(document_class, document_properties)
        self.usepakage(packages)
        self.file_header()
        self.write(self.header)

    def document_class(self, document_class, document_properties):
        options = ["[" + str(dp) + "]" for dp in document_properties]
        options = ''.join(options)
        self.dc = '\\documentclass' + options + '{' + document_class + '}\n'

    def usepakage(self, packages):
        up = ['\\usepackage{' + str(p) + '}\n' for p in packages]
        self.up = ''.join(up)

    def geometry(self, left='2cm', right='2cm', top='1cm', bottom='2cm', paper='a4paper'):
        x = 'geometry'
        y = 'left=' + str(left) + ',right=' + str(right) + ',top=' + str(top) + ',bottom=' + str(
            bottom) + ',paper=' + str(paper)
        self.write(self.command(x, y))

    def file_header(self):
        header = ''.join([self.dc, self.up])
        self.header = header

    def write(self, x):
        tabs = '\t' * self.tab
        self.file.write(tabs+x)
        if self.newline:
            self.file.write('\n')

    def save_tex(self):#
        temp_tex = 'tempfile.tex'
        f = open(temp_tex, 'w')
        f.write(self.file.getvalue())
        f.close()


    def compile(self):
        os.system("pdflatex tempfile.tex #> /dev/null 2>&1")

    def title(self, title):
        self.write('\\title{'+title+'}\n')

    def date(self, date='\\today'):
        self.write('\\date{'+date+'}\n')

    def author(self, author='T.H. Parry-Williams'):
        self.write('\\author{'+author+'}\n')

    def begin_document(self):
        self.write('\\begin{document}\n')

    def end_document(self):
        self.write('\\end{document}\n')

    def make_title(self):
        self.write(self.command('maketitle'))

    def section(self, x):
        self.write('\\section{'+str(x)+'}\n')

    def subsection(self, x):
        self.write('\\subsection{'+str(x)+'}\n')

    def subsubsection(self, x):
        self.write('\\subsubsection{'+str(x)+'}\n')

    def begin(self, x, options=None, options1=None):
        if options is not None:
            option = ['['+o+']' for o in options]
            option = ''.join(option)

        if options1 is not None:
            option1 = '{' + ''.join(options1) + '}'

        if options is not None:
            if options1 is not None:
                self.write('\\begin{'+str(x)+'}'+option+option1+'\n')
            else:
                self.write('\\begin{'+str(x)+'}'+option+'\n')
        else:
            if options1 is not None:
                self.write('\\begin{'+str(x)+'}'+option1+'\n')
            else:
                self.write('\\begin{'+str(x)+'}\n')

    def end(self, x):
        self.write('\\end{'+str(x)+'}\n')

    def newpage(self):
        self.write(self.command('newpage'))

    @staticmethod
    def command(x, y=None):
        if y is None:
            x = '\\' + x + '\n'
        else:
            x = '\\' + x + '{' + y + '}\n'
        return x

    def includegraphics(self, x, width, newline=True):
        command = '\\includegraphics'
        w = '[width=' + str(width) + ']'
        if newline:
            command = command + w + '{' + str(x) + '}\n'
        else:
            command = command + w + '{' + str(x) + '}'
        self.write(command)

    def figure(self, arg, centering=True, position='h', width='\\textwidth', caption=''):

        self.begin('figure', options=position)
        self.tab = 1

        if centering:
            self.write(self.command('centering'))

        self.includegraphics(arg, width=width, newline=False)

        if caption != '':
            self.write(self.command('caption', caption))

        self.tab = 0
        self.end('figure')

    def subfigure(self, *args, centering=True, position='h', caption='', width=0.5):
        self.newline = False
        self.begin('figure', options=['H'])
        self.tab = 1
        self.write(self.command('centering'))
        for i in range(len(args)):
            self.tab = 1
            self.begin('subfigure', options1=str(width) + '\\textwidth')
            self.tab = 2
            if centering:
                self.write(self.command('centering'))
                self.includegraphics(args[i], width='1\\linewidth', newline=True)
                self.newline = False
            self.tab = 1
            self.end('subfigure')
            self.write(self.command('hfill'))
            self.tab = 0
        if caption != '':
            self.write(self.command('caption', caption))
        self.end('figure')
        self.newline = True

    def table(self, df, centering=True, position='H', caption='', highlight=False):
        # if centering:
        # self.write(self.command('centering'))
        col_num = len(df.columns)
        col_format = 'l' + 'c' * (col_num)
        if highlight:
            df = df.style.background_gradient(cmap='RdYlGn_r', subset='MAPE (\%)', vmin=0, vmax=100).hide(axis="index")
        else:
            df = df.style.hide(axis="index")
        if caption != '':
            self.write(
                df.to_latex(column_format=col_format, caption=caption, position=position, position_float='centering',
                            hrules=True, convert_css=True))
        else:
            self.write(df.to_latex(column_format=col_format, position=position, position_float='centering', hrules=True,
                                   convert_css=True))
