# Simple-Traceability-SRS-Documents
Author  | [Rakha Asyrofi](https://scholar.google.com/citations?user=WN9T5UUAAAAJ&hl=id&oi=ao)
 -------|-----------
Version | 0.0.2
Updated | 2021-09-12

this is repository how to make simple traceability from requirement documents, 
you can check this out to [live demo](https://wordembed.herokuapp.com/)

Comparative Studies of Several Methods for Building Simple Traceability and Identifying The Quality Aspects of Requirements in SRS Documents described in [our paper at EECCIS2020](https://ieeexplore.ieee.org/document/9263479). Please kindly cite the following paper when you use this tool. It would also be appreciated if you send me a courtesy [website](http://rakha.asyrofi.com/) and [google scholar](https://scholar.google.com/citations?user=WN9T5UUAAAAJ&hl=id&oi=ao), so I could survey what kind of tasks the tool is used for. 
```
@inproceedings{asyrofi-2020-Traceability,
    title = "Comparative Studies of Several Methods for Building Simple Traceability and Identifying The Quality Aspects of Requirements in SRS Documents",
    author = "Asyrofi, Rakha  and
      Hidayat, Taufik and
      Rochimah, Siti",
    booktitle = "2020 10th Electrical Power, Electronics, Communications, Controls and Informatics Seminar (EECCIS)",
    month = November,
    year = "2020",
    address = "Malang, Indonesia",
    publisher = "IEEE",
    url = "https://doi.org/10.23919/EECSI50503.2020.9251905",
    pages = "243-247",
    language = "English",
    ISBN = "978-1-7281-7110-4",
}
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



