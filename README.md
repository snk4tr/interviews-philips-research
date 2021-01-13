# Technical screen at Philips Research Russia

This repo contains guidelines, questions and code snippets for the first stage of technical interviews - technical screen. 
Provided materials are intended to initiate deep technical discussions and help to understand candidate's level of
tehcnical expertise.

## Summary

- [Interview plan](#interview-plan)
    - [Introductory section](#introductory-section)
    - [Warm up section](#warm-up-section)
    - [Code review section 1](#code-review-section-1---shoot-in-your-leg)
    - [Code review section 2](#code-review-section-2---quantization-of-cnns)
    - [Wrap up section](#wrap-up-section)
- [Usage](#usage)
- [Tasks](#tasks)
- [Legend](#legend)

## Interview plan

Technical interview is conducted by two interviewers. Overall estimated time: 60-90 mins.  

#### Introductory section
**Goal**: <ins>greeting</ins> of a candidate and introductory words about the interview process. On this stage it is important to <ins>inform</ins>
the candidate about the whole <ins>interview porcess</ins> and <ins>notify</ins> that they will need to <ins>share their screen</ins> during the code review stage.  
**Estimated time**: 3-5 mins.

#### Warm up section
**Goal**: <ins>overcome</ins> possible <ins>anxiety</ins> and <ins>kick off a technical conversation</ins>, indentify weak spots in candidate's technical
experise.  
**Process**: ask several question from [the list (task 0)](docs/task0.md). Keep questions open-ended, 
let the candidate to talk more to warm up and feel more confident. 
Ask additional `why` and `how` questions if there is a feeling that the candidate is not providing explanations themselves.  
**Estimated time**: 10-15 mins.

#### Code review section 1 - Shoot in Your Leg
**Goal**: check a <ins>basic</ins> understanding of the principles of neural networks, 
ask questions <ins>"in depth"</ins> regarding the mechanisms of their work.  
**Process**: request the candidate to find mistakes in the [code snippet (task 2)](task2.py) that contains them. 
Ask additional `why` and `how` questions to verify that the candidate understands underlying principles and mechanisms,
which stand behind functions calls. Use the [reference](docs/task2.md) to find the full list of errors and more ideas of 
quesions for the candidate.   
**Estimated time**: 15-20 mins.

#### Code review section 2 - Quantization of CNNs
**Goal**: check deeper knowledge in the structure of neural networks, candidate's awareness of modern architectures, 
underlying principles of designing of such networks, talk about the technical aspects of deep learning at a higher 
and broader level than in the previous assignment.    
**Process**: let the candidate to skim thought [the code (task 1)](task1.py) for 2-3 mins and then start a technical discussion 
by asking questions on architecture design, data loading and other aspects that are a part of the code under review. 
Use the [reference](docs/task1.md) to find the list of possible Q&As. At the end of the [reference](docs/task1.md) 
there is a codind task. Request the candidate to implement the proposed functionality if there is doubt about 
the candidate's practical skills.      
**Estimated time**: 30-40 mins.

#### Wrap up section
**Goal**: let the candidate to ask their questions.  
**Estimated time**: 5-10 mins.


## Usage

Provide a candidate a code snippet that is repated to one of tasks of interest. Interviewing process includes but not limited to three types of conversations:

1. Going into details of implemetation using questions prepared in [docs](docs/). Please note that senctions in tasks are enumerated in order to ease the process of finding relevant questions.
2. Gereneral conversation on code's strengths and weaknesses. Discussion of possible solution for reimplementing it.
3. Actual live refactoring/reimplementation with the goal of meeting some formal requirements (production ready system, scalability etc.).

### Tasks

1. 📗 [**General questions**](docs/task0.md). These questions are supposed to be an easy way to kick off a technical conversation on the initial stage of the technical interview. 
2. 📗 [**Quantization of a neural network**](task1.py). Based on [PyTorch tutorial](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html).
3. 📗 [**Shoot in your leg**](task2.py) - compilation of basic mistakes that can be made when designing NN architecture (from Yandex Data School)
4. 📙 [**Modified CycleGAN**](task3.py). Based on [paper](https://arxiv.org/abs/1703.10593). Modified for the task of generation of realistic avatars from portrait photos of people.

### Legend

* 📗 - easy (junior scientist)
* 📙 - medium (scientist)
* 📕 - hard (senior scientist)
