import numpy as np

def table(x,y,name,title):
    J = len(a)
    Nx = 10
    length = "1.1cm"
    Ny = int(np.ceil(J/Nx))

    q = """
    \\begin{table}[htdp]"""
    for l in range(Ny-1):
        q+="""
        \\begin{tabular}{|l|%s|}
        \\hline"""%(("|p{%s}"%length)*Nx)

        if l==0: 
            q+="""
            \\multicolumn{%d}{|c|}{\\cellcolor[RGB]{206,250,201}$
            \\mathbf{%s}$} \\\\\n"""%(Nx+1,title)

        q += ("\\textbf{%s}"%(name[0])+"& %.2f"*Nx+" \\\\\n")%(tuple(x[Nx*l:Nx*(l+1)]))
        q += ("\\textbf{%s}"%(name[1])+"& %.2f"*Nx+" \\\\\n")%(tuple(y[Nx*l:Nx*(l+1)]))
        q+="""
        \\hline
        \\end{tabular}"""

    # last one
    l = Ny-1

    rest = J%Nx 
    hinten = Nx - rest -1

    q+="""
    \\begin{tabular}{|l|%s|}
    \\hline"""%(("|p{%s}"%length)*rest)

    if l==0: 
        q+="""
        \\multicolumn{%d}{|c|}{\\cellcolor[RGB]{206,250,201}$
        \\mathbf{%s}$} \\\\\n"""%(rest+1,title)

    q += ("\\textbf{%s}"%(name[0])+"& %.2f"*rest+" \\\\\n")%(tuple(x[Nx*l:(Nx*l+rest)]))
    q += ("\\textbf{%s}"%(name[1])+"& %.2f"*rest+" \\\\\n")%(tuple(y[Nx*l:(Nx*l+rest)]))
    q+="""
    \\hline
    \\end{tabular}"""

    q+="""
    \\caption{}
    \\label{Power05}
    \\end{table}"""
    print(q)


input_dir = "data_faraday/"

for i in range(4):
    a = np.load(input_dir +"a_%i.npy"%(i+1))
    I = np.load(input_dir + "i_%i.npy"%(i+1))
    names = ["angle $\\alpha$", "Current $I$"]
    table(a,I,names,"Measurement \quad 2.%d"%(i+1))

