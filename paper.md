---
title: 'ORDerly: Reaction cleaning of USPTO from ORD'
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

Availability of good data is crucial to the development of computational methods; however, open access to high quality chemical reaction data is severely lacking [Baldi:2022]. Within organic chemistry the largest open access data set is a collection of organic chemistry reactions extracted from patents, curtosey of the United States Patent and Trademark Office, and this data set has hence been dubbed "USPTO". One reason for the lack of high quality data is due to the fact that much of chemical data is reported in ways that are not easily machine interpretable (e.g. as tables in a pdf). The Open Reaction Database (ORD) was proposed as a solution to this problem, and has been proposed to be the standardised way of storing chemical reaction data in a standardised way. However, this data is still very noisy, hence the need for cleaning. ORDerly uses a range of computational checks & chemistry domain knowledge to validate and clean the data, to return a pandas DataFrame that will make clean chemistry data more accessible.



# Statement of need

`ORDerly` is a Python pakcage for cleaning reaction data stored in the format of Open Reaction Database (ORD)[@Kearnes:2021]. ORD is an open access chemical reaction database where data is stored in a standardised way (i.e. in accordance with the ORD schema). Currenlty, roughly 1.7 million reactions are stored in the ORD databse, most of which originating from data scraping of patents from the United States Patent and Trademark Office (USPTO), however, significant effort and domain knowledge is required to transform a set of raw reactions full of noise into useful data which is usable e.g. for training machine learning models. The type of model being built will often influence how cleaning is performed and which data consistency checks are run, making it difficult to re-use code written for other projects. This has led to cleaning of the USPTO dataset being a task that has been needlessly repeated many times over throughout the computational chemistry community(add citations?). The two most related works to this are [@molecularai_reaction_utils:2022] and [@RXN_reaction_preprocessing:2022]; the advantages of this work are (...)

The main functions carried out by this work are:

(...)

# Acknowledgements

This work is co-funded by UCB Pharma and Engineering and Physical Sciences Research Council via project EP/S024220/1 EPSRC Centre for Doctoral Training in Automated Chemical Synthesis Enabled by Digital Molecular Technologies.

# References