- M. Beekenkamp

# An Investigation into the Potential Sentiment Disparity in Earnings Calls between Prepared and Unprepared Sections.

## Introduction

Every quarter investors eagerly await company's earnings calls to hear management's perspective on a company's business performance. These earnings calls typically consist of two sections: a presentation, which is a prepared speech by management, and a Q&A section, during which management need to answer select shareholder questions. 

Acknowledging the importance of these transcripts, companies spend a lot of time ensuring that the presentation is precisely worded to convey the company's health in the correct tone, however, during the Q&A section questions need to be answered in real time and are perhaps more revealing to the true state of the company. 

So, perhaps there is an information advantage to be exploited by performing sentiment analysis on each segment separately potentially exposing 'truer' beliefs. To begin with I want to look at the sentiment disparity between the two sections, but this project can be expanded to see which section is more illuminating, or if the transcript should be digested as a whole for the most accurate interpretation.

## Use Guide

This code was written in Python 3.11.4, and uses the following packages:

- tensorflow
- numpy
- scipy
- pandas
- matplotlib
- scikit-learn
- nltk

>[!TIP]
>If you don't have these packages installed, refer to the **installation guide below** for a bash script that will install all of these packages.

<details>
<summary>Installation Guide</summary>

### Installation

To install the required packages, you will need Python 3.8–3.11 and pip version >19.0 (or >20.3 for macOS) installed. For mac users on Apple Silicon, you will also need to install the `tensorflow-macos` package so that the code can run on GPU.

To check if you have a compatible version of Python installed, run the following command in the terminal:

```bash
python --version
```

And to update pip, run the following command in the terminal:

```bash
pip install --upgrade pip
```

Then to install packages, run the following bash script:

```bash
chmod +x install_packages.sh
./install_packages.sh
```

</details>

### Running the code

To run the code, you have two options. You can either specify the company you want to analyse, or you can run the code on all companies. To run the code on all companies, you can run the following command in the terminal:

```bash
python main.py
```

However, if you have a specific company you want to analyse, you can simply run the following command in the terminal:

```bash
python main.py --company
```

After which you will see `>> ` in the terminal, denoting that you are in the REPL.

You can then enter the symbol of the company you want to analyse. For the dataset used in this project, the company symbol's are as follows:
- AAPL (Apple)
- AMD (Advanced Micro Devices)
- AMZN (Amazon)
- ASML (ASML Holding)
- CSCO (Cisco)
- GOOGL (Alphabet)
- INTC (Intel)
- MSFT (Microsoft)
- MU (Micron Technology)
- NVDA (Nvidia)

By entering the company's ticker symbol, you can analyse the sentiment of the earnings call for that company. After running the code, you will have the opportunity to enter another company to analyse, or to exit the program.

To exit the program at any time, you can simply type `:exit` inside the REPL.

## Data Sources and Descriptions

All data is downloaded from a [kaggle data set](https://www.kaggle.com/datasets/ashwinm500/earnings-call-transcripts/code). The data contains Quarterly Earnings Call Transcripts for 10 NASDAQ companies from 2016-2020 in the form plain text files (.txt).

TODO: Will likely use BERT or another pre-trained model alongside my own sentiment analysis model to compare the two. 

## Method

TODO: Explain the method used to analyse the data.

This projects includes three main methods of analysis:
1. Document Similarity Analysis
    - This section compares the similarity between the prepared presentation and the Q&A section of the earnings call for each company. This is achieved via TF-IDF and cosine similarity.
2. Transformer Sentiment Analysis
    - This section uses a pre-trained transformer model to generate sentiment labels for the prepared presentation and Q&A section of the earnings call for each company. 
    - These are used as the training labels for the RNN, LSTM, and GRU models. Then the transformer models are used to generate new sentiment labels.
3. Pre-trained Sentiment Analysis
    - This section uses a pre-trained sentiment analysis model as a starting point, and then fine-tunes the model for our dataset.

A limitation in the project is that in the Q&A section I haven't isolated the answers from the questions, so the analysis will contain some noise from the questions. This is because there is no reliable formatting convention that would allow me to systematically separate the two. However, considering that the questions are asked by shareholders and ultimately this tool is for investors, the sentiment of the questions is perhaps not purely noise and could be useful in part.

### Maths Review

<details>
<summary>TF-IDF</summary>
#### TF-IDF

Term Frequency-Inverse Document Frequency (TF-IDF) is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. It is often used as a weighting factor in information retrieval and text mining. The TF-IDF value increases proportionally to the number of times a word appears in the document and is offset by the number of documents in the corpus that contain the word, which helps to adjust for the fact that some words appear more frequently in general.

The formula for TF-IDF is as follows:

$$ w_{i,j} = tf_{i,j} \times log(\frac{N}{df_i}) $$

Where:
- $w_{i,j}$ is the TF-IDF weight of term $i$ in document $j$;
- $tf_{i,j}$ is the term frequency of term $i$ in document $j$;
- $N$ is the total number of documents in the corpus;
- $df_i$ is the number of documents in the corpus that contain term $i$;

</details>

<details>
<summary>Cosine Similarity</summary>
#### Cosine Similarity

Cosine similarity is a metric used to determine how similar two vectors are. In this case, the vectors are the TF-IDF vectors of the prepared presentation and Q&A section of the earnings call for each company. Cosine similarity measures the cosine of the angle between two vectors projected in a multi-dimensional space. The formula for cosine similarity is as follows:

$$ \text{cosine similarity} = \frac{\textbf{A} \cdot \textbf{B}}{\|\textbf{A}\| \times \|\textbf{B}\|} $$

Where:
- $\textbf{A}$ is the TF-IDF matrix for the presentation;
- $\textbf{B}$ is the TF-IDF matrix for the Q&A section;

</details>


#### RNNs

A Recurrent Neural Network (RNN) is a type of neural network that is designed to handle sequential data. It is particularly useful for natural language processing tasks, as it can remember information from previous time steps and use it to make predictions at the current time step. The two main types of RNNs are the LSTM and GRU networks.

<details>
<summary>LSTM</summary>
#### LSTM

A Long Short-Term Memory (LSTM) network is a type of RNN that is designed to handle long-term dependencies in sequential data. It is particularly useful for natural language processing tasks, as it can remember information from previous time steps and use it to make predictions at the current time step. The basic structure of an LSTM is as follows:

TODO: Insert formula / Insert Diagram

</details>

<details>
<summary>GRU</summary>
#### GRU

A Gated Recurrent Unit (GRU) network is a type of RNN that is designed to handle long-term dependencies in sequential data. It is particularly useful for natural language processing tasks, as it can remember information from previous time steps and use it to make predictions at the current time step. The basic structure of a GRU is as follows:

TODO: Insert formula / Insert Diagram
Paper: https://arxiv.org/pdf/1412.3555.pdf

</details>

## Results

TODO: Explain the results of the analysis.

## Conclusion

TODO: Summarise the findings and suggest future expansion.

## References

`Data/Dataset/`: <br>
[Quarterly Earnings Call Transcripts for 10 NASDAQ companies from 2016-2020](https://www.kaggle.com/datasets/ashwinm500/earnings-call-transcripts/code) (released: 07/03/2024) <br><br>
