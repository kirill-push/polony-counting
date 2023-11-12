[![Linting](https://github.com/kirill-push/polony-counting/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/kirill-push/polony-counting/actions/workflows/lint.yml)

## Project Description
**Polony Counter**

Polony Counter is a protocol originally developed based on PCR for sequencing purposes. It has been further refined and adapted by Lindell's Lab at Technion (Haifa, Israel) for the specific task of quantifying viruses in marine water at the type/clade level. This means it goes beyond just counting "how many viruses" and provides precise information on "how many viruses of a particular type" are present in a given sample.

### Overview

Analyzing 20-30 samples of marine water using the Polony Counter protocol typically takes 2.5 days. The process is divided into two main phases:

**Molecular Biology Work**: This phase, which spans two days, is conducted in the laboratory. It involves various molecular biology procedures to prepare samples for analysis.

**Polony Counting**: After completing the molecular biology work, high-resolution images of slides are captured using a scanner. These images include one RGB image and two black-and-white images representing the green and red channels, respectively. The slides contain various points known as "polonies" that appear in red, green, and yellow colors.

### Project Goals

The primary objective of this project is to streamline and automate the counting process of these polonies. The aim is to reduce the amount of time researchers spend on manual counting, which can be labor-intensive and time-consuming. By automating this aspect of the protocol, the project will significantly increase the efficiency and accuracy of virus quantification in marine water samples.

## Project Organization

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── config             <- Configuration files for the project.
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── make_dataset.py
    │   │   └── utils.py
    │   │
    │   └── models         <- Scripts with models
    │       ├── models.py
    │       ├── predict_model.py
    │       ├── train_model.py
    │       └── utils.py
    │
    └── tests              <- Scripts for functions and module testing

--------

<!-- ## Getting Started:-->

### Contributions

Contributions to this project are welcome, and developers, scientists, and researchers are encouraged to participate in its development. Whether you have expertise in image processing, machine learning, or molecular biology, your contributions can help advance this valuable tool for virus quantification in marine water samples.


### Feedback:

I value feedback from the community. If you encounter any issues, have suggestions for improvements, or would like to report bugs, please don't hesitate to open an issue on the GitHub repository.

### Acknowledgments

I would like to express my gratitude to Lindell's Lab at Technion for their pioneering work on the Polony Counter protocol, which serves as the foundation for this project.

### License

This project is open-source and released under the MIT License. Please refer to the project's LICENSE file for more details on licensing.
<!-- 
### About the Project

This project is a labor of love and a volunteer effort by me. I have a deep passion for advancing science and simplifying the work of researchers in the field. As a result, I have dedicated my time and expertise to develop and maintain the Polony Counter project with no financial compensation.

**Key Points:**

- 🌟 Voluntary Contribution: This project is developed on a voluntary basis, driven by a commitment to contribute to the scientific community.

- 💪 Sole Developer: I am the sole developer of this project, working tirelessly to make it accessible and useful for researchers worldwide.

- 🆓 No Cost: The Polony Counter project is completely free and open-source, with no charges for its usage. -->

