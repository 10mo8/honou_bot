from glob import glob
import re

dir_name = "./2012data"
fnamelst = ["abe", "asou", "hukuyama", "inada", "mizuho", "rennhou"]

for i, file in enumerate(glob(dir_name + '/*.txt')):
    with open(file, encoding="utf-8") as fin, open("./2012data/" + fnamelst[i] + ".txt" , "w", encoding="utf-8") as fout:
        texts = fin.readlines()
        print(len(texts))
        for text in texts:
            en_text, ja_text = text.strip().split(',')[:2]

            en_text = re.sub('^ ', '', en_text)
            en_text = re.sub('$ ', '', en_text)
              
            ja_text = re.sub("^ ", "", ja_text)
            ja_text = re.sub('$ ', '', ja_text)
            
            fout.write(en_text[:-1] + "," + ja_text + "\n")

