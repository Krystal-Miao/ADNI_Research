# ADNI MRI Preprocessing Pipeline

## Overview

This repository contains the preprocessing pipeline developed for Alzheimer's Disease (AD) versus Cognitively Normal (CN) classification using T1-weighted MRI scans from the ADNI dataset.

## Project Objectives

- Construct an AD/CN cohort from ADNI1.
- Filter baseline MRI scans.
- Select one scan per subject.
- Validate MRI data quality.
- Reproduce an FSL-based preprocessing pipeline.
- Prepare data for future deep learning experiments.

## Current Cohort

| Group | Subjects |
|---------|---------|
| CN | 151 |
| AD | 97 |
| Total | 248 |

## Preprocessing Pipeline

1. Reorientation (FSL)
2. Cropping (FSL)
3. Brain Extraction (BET)
4. Registration to MNI152 (FLIRT)
5. Bias Field Correction (FAST)
6. Intensity Normalization
7. Central Slice Selection
8. TIFF Export

## Repository Structure

- scripts/ : cohort construction and quality-control scripts
- preprocessing/ : MRI preprocessing pipeline
- reports/ : progress reports and documentation

## Notes

ADNI data are not included in this repository due to data usage restrictions.