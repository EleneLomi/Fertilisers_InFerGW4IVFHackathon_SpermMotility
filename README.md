# Fertilisers
This repository is a group entry to "Fertility: In Vitro, In Silico, In Clinico" hackathon.


## <span style="color:blue">GOAL: We need to choose the project.</span>

## Table of Contents

- [Fertilisers](#fertilisers)
  - [GOAL: We need to choose the project.](#goal-we-need-to-choose-the-project)
  - [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Background](#background)
- [Path Extraction](#path-extraction)
  - [Motivation and Summary](#motivation-and-summary)
  - [Alogirithm and Implementation.](#alogirithm-and-implementation)
  - [Benchmarking](#benchmarking)
- [Path Analysis](#path-analysis)
- [Team](#team)

---
# Introduction

# Background

# Path Extraction
## Motivation and Summary
The data provided for this challenge is pre-tracked videos from 2 sperm samples moving in vitro. To analyse the motion of the sperm we first need to extract the path the sperm takes from the videos. To do this we use the Lucas-Kanade method to estimate the background movement velocity at several "corner" points, take the average velocity after removing outliers, and use the average velocity to build up a path. We went on to validate this method qualitatively using overlaid path animations and quantiatively against hand tracked data, and saw high accuracy. The method is performant, running ~ 1 frame per 0.1ms, and so could easily be adapted to run with a live video stream in real time. Although the real world applicability of this path extraction method in the IVF setting may be slightly limited as it seems likely the system that initally tracked the sperm would record the path data as well, it is plausible that the path data may be lost in a data wipe or hard to accesss in propeitary software and a method such as this one would become necessary. 
## Alogirithm and Implementation.
- Every iteration choose good points to track, excluding points in the center.
- Use Lucas-Kanade method to estimate the local optical flow at a number of points 
- Remove outliers using the mahalanobis distance
- Average the remaining flow vectors
- (Smoothing using kalman filter?)
- Update the path 
## Benchmarking
- Accuracy against hand tracked 
- Performance

# Path Analysis

# Team
Our team is made up of Mitja Devetak, Elene Lominadze, Ben Nicholls-Mindlin and Peter Waldert. We all met studying Mathematical Modelling and Scientific Computing at the University of Oxford.