# Technical screen at Philips Research Russia

This repo contains guidelines, questions, and code snippets for the first technical interview stage - the technical screen.
Provided materials are intended to initiate in-depth technical discussions and help to understand the candidate's level of
technical expertise in Deep Learning-based Computer Vision.

## Summary

- [Interview plan](#interview-plan)
    - [Introduction](#introduction)
    - [Warm up section](#warm-up-section)
    - [Code review section 1](#code-review-section-1---shoot-in-your-leg)
    - [Code review section 2](#code-review-section-2---quantization-of-cnns)
    - [[OPTIONAL] Code review section 3](#optional-code-review-section-3---modified-cyclegan)
    - [Wrap up section](#wrap-up-section)
- [Usage](#usage)
- [Tasks](#tasks)
- [Legend](#legend)
- [Contacts](#constacts)

## Interview plan

Two or more interviewers conduct the technical interview.  Interviewers use this repository as a reference.

Interviewers are free to substitute tasks as they see fit, although the following order is preferred:   
indroduction -> warm up -> code review secion 1 -> code review secion 2 -> code review section 3 (optional) -> wrap up.

Overall estimated time: 60-90 mins.

#### Introduction
**Goal**: <ins>greeting</ins> of a candidate and introductory words about the interview process. On this stage it is 
important to <ins>inform</ins> the candidate about the whole <ins>interview porcess</ins> and <ins>notify</ins> 
that they will need to <ins>share their screen</ins> during the code review stage.  
**Estimated time**: 3-5 mins.

#### Warm up section
**Goal**: <ins>overcome</ins> possible <ins>anxiety</ins> and <ins>kick off a technical conversation</ins>, indentify 
weak spots in candidate's technical
experise.  
**Process**: ask several question from [the list (task 0)](docs/task0.md). Keep questions open-ended,
let the candidate to talk more to warm up and feel more confident.
Ask additional `why` and `how` questions if there is a feeling that the candidate is not providing explanations themselves.  
**Estimated time**: 10-15 mins.

#### Code review section 1 - Shoot in Your Leg
**Goal**: check a <ins>basic</ins> understanding of the principles of neural networks,
ask questions <ins>"in-depth"</ins> regarding the mechanisms of their work.  
**Process**: request the candidate to find mistakes in the [code snippet (task 2)](task2.py) that contains them.
Ask additional `why` and `how` questions to verify that the candidate understands underlying principles and mechanisms,
which stand behind function calls. Use the [reference](docs/task2.md) to find the full list of errors and more ideas of 
questions for the candidate.   
**Estimated time**: 15-20 mins.

#### Code review section 2 - Quantization of CNNs
**Goal**: check deeper knowledge in the structure of neural networks, candidate's awareness of modern architectures,
underlying principles of designing of such networks, talk about the technical aspects of deep learning at a higher and 
broader level than in the previous assignment.    
**Process**: let the candidate skim through [the code (task 1)](task1.py) for 2-3 mins and then start a technical 
discussion by asking questions on architecture design, data loading, and other aspects that are a part of the code under review.
Use the [reference](docs/task1.md) to find the list of possible Q&As. At the end of the [reference](docs/task1.md)
there is a coding task. Request the candidate to implement the proposed functionality if there is doubt about the 
candidate's practical skills.      
**Estimated time**: 30-40 mins.

#### [OPTIONAL] Code review section 3 - Modified CycleGAN
**Goal**: check in-depth knowledge in one of the advanced areas of machine learning research -
Generative Adversarial Networks (GANs). This task is designed for senior-level candidates and intended to verify both 
their theoretical and practical skills in designing and training GANs.  
**Process**: let the candidate skim through [the code (task 3)](task3.py) for 3-5 mins and then start a technical 
discussion by asking questions on architecture design, data loading, and other aspects that are a part of the code
under review. Use the [reference](docs/task3.md) to find the list of possible Q&As.  
**Estimated time**: 30-40 mins.

#### Wrap up section
**Goal**: let the candidate to ask their questions, say goodbye to the interviewee.  
**Estimated time**: 5-10 mins.


## Usage


During code review sections, [repository with code for a candidate](https://github.com/snk4tr/interviews-philips-research-candidate)
is shared with the candidate by changing its visibility to `public` and sending them a link. If an interviewer does not
have access to the [repository with code for a candidate](https://github.com/snk4tr/interviews-philips-research-candidate),
they should contact <ins>sergey.kastryulin@philips.com</ins> and request it.
The [repository with code for a candidate](https://github.com/snk4tr/interviews-philips-research-candidate) contains
the same code snippets except for some inline comments, which, if present, would provide unnecessary extra support for
the candidate. After the code review process is finished, the interviewer who shared the link to the candidate's
repository has to change its visibility back to `private` to limit unnecessary public access.

Interviewers are encourage but not limited to the following actions:
1. Going into details of implementation using questions prepared in [docs](docs/).
   Please note that sections in tasks are enumerated to ease the process of finding relevant questions.
2. General conversation on the code's strengths and weaknesses. Discussion of possible solutions and/or 
   reimplementations of some parts of it.
3. Live refactoring/reimplementation/coding to meet some formal requirements (production-ready system, scalability, etc.).

### Tasks

1. 📗 [**General questions**](docs/task0.md). These questions are supposed to be an easy way to kick off a technical 
   conversation in the initial stage of the technical interview.
2. 📙 [**Quantization of a neural network**](task1.py). Based on 
   [PyTorch tutorial](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html).
3. 📗 [**Shoot in your leg**](task2.py) - a compilation of basic mistakes that can be made when designing NN 
   architecture (from Yandex Data School)
4. 📕 [**Modified CycleGAN**](task3.py). Based on [paper](https://arxiv.org/abs/1703.10593). Modified for the task of 
   a generation of realistic avatars from portrait photos of people.

### Legend

* 📗 - easy (junior scientist)
* 📙 - medium (junior scientist+ / scientist)
* 📕 - hard (scientist+ / senior scientist)

### Constacts

In case of any questions regarding the interview process please reach out to:
- Sergey Kastryulin ([snk4tr](https://github.com/snk4tr)) - <ins>sergey.kastryulin@philips.com</ins>
- Irina Fedulova ([irifed](https://github.com/irifed)) - <ins>irina.fedulova@philips.com</ins>
