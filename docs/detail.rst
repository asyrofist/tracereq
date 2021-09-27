Comparative Studies of Several Methods for Building Simple Traceability and Identifying The Quality Aspects of Requirements in SRS Documents
=========================================================================================

Overview
------------

Secara garis besar, library ini dibuat untuk mengembangkan metode ekstraksi kebergantungan kebutuhan menggunakan pemrosesaan bahasa alamiah, 
yang telah diterangkan pada proceeding conference di  `EECCIS2021`_. 
Jika anda menggunakan library ini, saya sangat mengapresiasi, dengan cara mengirimkan segala macam bentuk kiriman melalui `courtesy`_  dan `scholar`_, 
Semoga data yang saya publikasikan, berguna untuk orang banyak, terima kasih. 

Abstrak
------------
In the software development process, the requirements traceability in the Software Requirements Specification (SRS) is an important aspect to trace fulfillment of requirement. Generally, the implementation of traceability is done manually, and it requires considerable time and money. Many methods have been proposed to identify traceability automatically, therefore, we conducted a comparative study of several existing methods to find out which method is better. In this study, we made a comparison of 4 methods, namely the Information Retrieval (IR), Ontology, Combination of Information Retrieval with Topic Modeling (LSA or LDA) methods. From the comparison of those methods, we found that the IR method gets the highest accuracy score, but the precision, recall and fl-score are unequal between dataset 1 and dataset 2, while other methods (ontology, IR+LSA, IR+LDA) have more stable metrics. The average score of precision from highest to lowest consecutively is in the IR + LDA method, Ontology, then IR+LSA.

.. _EECCIS2021: https://ieeexplore.ieee.org/document/9263479
.. _courtesy: https://www.researchgate.net/profile/Rakha_Asyrofi
.. _scholar: https://scholar.google.com/citations?user=WN9T5UUAAAAJ&hl=id&oi=ao

Dikembangkan oleh Rakha Asyrofi (c) 2021

Cara menginstall
--------------

Instalasi menggunakan PYPI:

    pip install tracereq

Fitur yang digunakan
------------
Berikut ini adalah beberapa fitur yang telah digunakan sebagai berikut:

- library ini dapat melacak kebergantungan kebutuhan

Kontribusi
------------
Sebagai bahan pengemabangan saya, maka saya apresiasi apabila anda, dapat mengecek issue dari repository library ini.

- Issue Tracker: https://github.com/asyrofist/Simple-Traceability-SRS-Document/issues
- Source Code: https://github.com/asyrofist/Simple-Traceability-SRS-Document

Support
------------
Jika anda memiliki masalah, saat menggunakan library ini. Mohon dapat membuat mailing list ke at: asyrofi.19051@mhs.its.ac.id

Lisensi
------------
Proyek ini dapat lisensi atas MIT License
