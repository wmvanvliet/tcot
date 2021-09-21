import sys
import colorama

from document import parse_document


colorama.init()
doc = parse_document(sys.argv[1])
doc.save()

for s in doc.sentences:
    if s.trust > 0:
        print(colorama.Fore.GREEN + str(s) + colorama.Style.RESET_ALL)
        print(colorama.Fore.GREEN + '\t' + str(s.references) + colorama.Style.RESET_ALL)
    elif s.garbled:
        print(colorama.Fore.YELLOW + str(s) + colorama.Style.RESET_ALL)
    elif s.trust < 0:
        print(colorama.Fore.RED + str(s) + colorama.Style.RESET_ALL)
    else:
        print(str(s))
