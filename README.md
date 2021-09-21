Text chain of trust
===================

Experimentation with a chain of trust system for sentences in academic papers.
Read more about it [here](https://hackmd.io/@wmvanvliet/ByrDbZD7F).

Author: Marijn van Vliet


Installation and usage
----------------------

Required packages: colorama, numpy, pytorch, pytorch-transformers, scipy.  
These are perhaps best installed through anaconda, but you can also try with PIP:

```
pip install -r requirements.txt
```

To run the analysis, first parse the XML for the papers and find sentence embeddings:

```
python parse_paper.py paper1.xml
python parse_paper.py paper2.xml
```

This will store the parsed version in a folder structure based on the DOI of the papers.
Finally, you can run the chain of trust verification process with:

```
python verify_paper.py 10.1016/j.neuroimage.2019.116221 10.1016/j.neuroimage.2013.10.067
```
