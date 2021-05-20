from os import listdir
from os.path import isfile, join

fp="../adclog/yes"
outdir='../tensor/'
word = 'yes'
zero_axis=950
max_range=120
samples=7812

output=0

with open(fp, 'r') as f:
    lines = f.readlines()
    loops = int(len(lines) / samples)
    for ll in range(loops):
        result = []
        for i in range(ll * samples, ll * samples + samples):
            line = lines[i]
            try:
                float(line)
            except ValueError:
                continue
            val=(float(line)-zero_axis)/max_range
            result.append(val)

        #search for word number
        onlyfiles = [f for f in listdir(outdir) if isfile(join(outdir, f))]
        searchn = 0
        for n in onlyfiles:
            if word in n:
                s = ''.join(x for x in n if x.isdigit())
            if s.isdigit():
                searchn = int(s) + 1
        with open(outdir+word+str(searchn), 'w') as fout:
            writestr = "\n".join(map(str, result))
            fout.write(writestr)

