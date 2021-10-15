"""Ini adalah indeks inisialiasi modul ini dimuat, terdiri atas prepreocessing, vsm, lda, dan lsa.
semua data tersimpan dengan baik, bisa dipelajar dengan cepat.
"""
from tracereq.lda import latentDirichlet
from tracereq.lsa import latentSemantic
from tracereq.preprocessing_evaluation import (pengukuranEvaluasi,
                                                   prosesData)
from tracereq.vsm import measurement
