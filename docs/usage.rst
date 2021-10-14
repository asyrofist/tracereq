============
Instalasi
============

Install langsung dari pypi
-------------------

Install the package with pip::

    $ pip install tracereq

Cara ini digunakan untuk pengguna dapat menggunakan library ini. adapun untuk menginstal dapat menggunakan cara lain, seperti langsung menginstal dari repository package library ini dari 

Install Langsung dari Repository
-------------------

Atau jika terlalu susah untuk menginstalnya, tinggal download berkas dari laman berikut ini. Cukup mudah dan simpel tinggal klik bagian code dan download zip tersebut.

**Download extract-req**: https://github.com/asyrofist/Simple-Traceability-SRS-Document

::

    cd Simple-Traceability-SRS-Documents
    python setup.py install
    # If root permission is not available, add --user command like below
    # python setup.py install --user

Currently, pip install is not supported but will be addressed.


Instalasi Keterlacakan Dokumen SKPL
------------------------------------------
Library ini dapat digunakan menggunakan spesifikasi dibawah ini, dimana python dan requirement yang dibutuhkan adalah sebagai berikut.
karena pengembangan menggunakan environment 3.7 maka disarankan untuk menginstal diatas versi tersebut.

- Python :code:`>= 3.7`

Requirements
------------
Dalam instalasi ini, membutuhkan package yang lain seperti daftar berikut ini. anda bisa melihatnya di 
(bagian depan repository github saya yang berada di :doc:`/requirement.txt` section.) 
Segala macam detail saya jelaskan pada sebagai berikut.

- numpy
- nltk
- matplotlib
- pandas
- xlrd
- sklearn
- py-automl


========
Penggunaan
========
Contoh Penggunaan Library
------------

Bagaimana cara menggunakan template ini, dapat dilihat dari contoh syntax berikut ini::

	from tracereq.preprocessing_evaluation import prosesData
	myProses = prosesData(inputData) #indeks data
	myProses.preprocessing()
	myProses.fulldataset(inputData)

Berikut ini penjelasan singkat darri contoh syntax tersebut.

- myPart.preprocessing()
bagian ini menunjukkan bagaimana cara pengembang melihat daftar dataset yang digunakan. Daftar dataset ini diambil dari excel dengan memilah daftar sheet yang digunakan. sehingga dengan jelas memperlihatkan daftar data yang digunakan.

- myPart.fulldataset(inputData) 
Bagian ini memperlihatkan dataset secara secara spesifik, sehingga cocok digunakan untuk data_raw awal sebelum dilakukan pra-pemrosesan maupun kegiatan lainnya. Karena data tersebut cenderung berbeda-beda terhadap setiap hasil yang diambil. 
