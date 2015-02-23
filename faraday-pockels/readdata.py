import numpy as np

def table(x,y,name,title):
    J = len(a)
    Nx = 10
    length = "1.1cm"
    Ny = int(np.ceil(J/Nx))

    q = """
    \\begin{table}[htdp]"""
    for l in range(Ny):
        endrange   =  Nx*(l+1)
        alpha = Nx
        if l==(Ny-1):
            alpha = J%Nx 
            endrange = Nx*l + alpha
        q+="""
        \\begin{tabular}{|l|%s|}
        \\hline"""%(("|p{%s}"%length)*alpha)
        if l==0: 
            q+="""
            \\multicolumn{%d}{|c|}{\\cellcolor[RGB]{206,250,201}$
            \\mathbf{%s}$} \\\\\n"""%(alpha+1,title)
        q += ("\\textbf{%s}"%(name[0])+"& %.2f"*alpha+" \\\\\n")%(tuple(x[Nx*l:endrange]))
        q += ("\\textbf{%s}"%(name[1])+"& %.2f"*alpha+" \\\\\n")%(tuple(y[Nx*l:endrange]))
        q+="""
        \\hline
        \\end{tabular}"""
    q+="""
    \\caption{}
    \\label{}
    \\end{table}"""
    print(q)

input_dir = "data_faraday/"

for i in range(1):
    a = np.load(input_dir +"a_%i.npy"%(i+1))
    I = np.load(input_dir + "i_%i.npy"%(i+1))
    names = ["angle $\\alpha$", "Current $I$"]
    table(a,I,names,"Measurement \quad 2.%d"%(i+1))

