---
title: 'ORDerly: A Python package for cleaning Open Reaction Database'
tags:
  - Python
  - Organic Reactions
  - Big Data Cleaning
authors:
  - name: Daniel S. Wigh
    orcid: 0000-0002-0494-643X
    affiliation: 1
  - name: Joe Arrowsmith
    affiliation: 2
  - name: Alexander Pomberger
    orcid: 0000-0003-2267-7090
    affiliation: 1
  - name: Alexei A. Lapkin
    orcid: 0000-0001-7621-0889
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
affiliations:
 - name: Department of Chemical Engineering and Biotechnology, University of Cambridge, Cambridge, UK
   index: 1
 - name: Independent Researcher, UK
   index: 2
date: 11 March 2023
bibliography: paper.bib

---

# Summary

Availability of good data is crucial to the development of computational methods; however, open access to high quality chemical reaction data is severely lacking [Baldi:2022]. Access to chemical reaction data is complicated by the fact that much of the data is not easily interpretable by computers (e.g. reported as tables in the PDFs of chemistry papers), proprietary, or locked behind a paywall. The Open Reaction Database (ORD) [@Kearnes:2021] is an open access chemical reaction database that stores reaction data in accordance with the ORD schema, and is proposed as a solution to this problem by providing a standardised way of storing raw chemical reaction data. 

# Statement of need

`ORDerly` is a Python package for cleaning chemical reaction data stored in ORD. ORD currently holds about 1.7 million reactions, most of which are reactions scraped from patent applications and grants with the United States Patent and Trademark Office (USPTO) [@Lowe:2016]. The USPTO dataset has been used in a number of papers within the machine learning for chemistry community [Predicting_Organic:2017,schwaller_mapping:2021,schneider_development:2015]. However, significant cleaning and filtering is required before any modeling can be done, and since the cleaning steps will vary depending on the prediction task it is likely that the task of cleaning USPTO data has been repeated numerous times by adjacent research groups within the community. Standardised approaches to cleaning of reactions have indeed been proposed before[@RXN_reaction_preprocessing:2020,@molecularai_reaction_utils:2022]. `ORDerly ` performs standard cleaning operations such as canonicalisation and duplicate removal as prior work, while also offer multiple advantages:
 1. Input is ORD, as opposed to raw USPTO data in XML, csv, or data already preprocessed to the point of being in a DataFrame in Python. This advantage will become especially relevant as ORD is more widely adopted for reporting of chemical data.
 2. Directed cleaning depending on the prediction task. Currently supported prediction modes: mapped reaction, mapped reaction + yield, mapped reaction + yield + conditions. A simpler prediciton mode will bypass certain cleaning steps, leading to a larger dataset.
 3. Reaction conditions and plain text descriptions are extracted. 
 4. Solvents identified using list of solvents, as opposed to relying on original labelling, with catalysts and reagents remapped to 'agents'.
 5. Filtering of reactions containing either too many components, or molecules with low frequency.
 6. Semi-manual name resolution of molecules not represented with SMILES.

`ORDerly` was designed to be the easiest way to obtain a cleaned dataset of the open access chemical reactions currenlty available, and the different cleaning modes available will allow researchers with different prediction goals to obtain a benchmarking dataset all from the same library.


# Acknowledgements

This work is co-funded by UCB Pharma and Engineering and Physical Sciences Research Council via project EP/S024220/1 EPSRC Centre for Doctoral Training in Automated Chemical Synthesis Enabled by Digital Molecular Technologies.

# References