[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5528405.svg)](https://doi.org/10.5281/zenodo.5528405)
![image](https://visitor-badge.laobi.icu/badge?page_id=asyrofist/Simple-Traceability-SRS-Document) 
![PyPI - Python Version](https://img.shields.io/badge/python-3.7.0-blue.svg)
[![PyPI](https://img.shields.io/pypi/v/tracereq.svg)](https://pypi.org/project/tracereq/)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)
[![Build Status](https://app.travis-ci.com/asyrofist/Simple-Traceability-SRS-Document.svg?branch=main)](https://app.travis-ci.com/asyrofist/Simple-Traceability-SRS-Document)
[![Documentation Status](https://readthedocs.org/projects/simple-tracebility/badge/?version=latest)](https://simple-tracebility.readthedocs.io/en/latest/?badge=latest)
[![Paper](http://img.shields.io/badge/Paper-PDF-red.svg)](https://ieeexplore.ieee.org/document/9263479)

# Simple-Traceability-SRS-Documents
Author  | [Rakha Asyrofi](https://scholar.google.com/citations?user=WN9T5UUAAAAJ&hl=id&oi=ao)
 -------|-----------
Version | 0.0.2
Updated | 2021-09-12

this is repository how to make simple traceability from requirement documents, 
you can check this out to [live demo](https://wordembed.herokuapp.com/)

Comparative Studies of Several Methods for Building Simple Traceability and Identifying The Quality Aspects of Requirements in SRS Documents described in [our paper at EECCIS2020](https://ieeexplore.ieee.org/document/9263479). Please kindly cite the following paper when you use this tool. It would also be appreciated if you send me a courtesy [website](http://rakha.asyrofi.com/) and [google scholar](https://scholar.google.com/citations?user=WN9T5UUAAAAJ&hl=id&oi=ao), so I could survey what kind of tasks the tool is used for. 
```
@INPROCEEDINGS{9263479,  author={Asyrofi, Rakha and Hidayat, Taufik and Rochimah, Siti},  
booktitle={2020 10th Electrical Power, Electronics, Communications, Controls and Informatics Seminar (EECCIS)},   
title={Comparative Studies of Several Methods for Building Simple Traceability and Identifying The Quality Aspects of Requirements in SRS Documents},   
year={2020},  
volume={},  
number={},  
pages={243-247},  
doi={10.1109/EECCIS49483.2020.9263479}}
```

Developed by Asyrofi (c) 2021

## Cara menginstal

instalasi melalui pypi:

    pip install tracereq


## Cara menggunakan program

```python
from tracereq.preprocessing_evaluation import prosesData
myProses = prosesData(inputData= 'dataset.xlsx')
myProses.preprocessing()
```

Check out: https://youtu.be/9FcAO_wbG_I



