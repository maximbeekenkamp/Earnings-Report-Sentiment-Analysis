import os
import pandas as pd
import re
import nltk 
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

class DataSet:
    def __init__(self, directory="Data/Dataset/Transcripts/", singleCompany=False):
        """
        Load the data from the directory and cleans it.
        This class also does some basic data cleaning.

        Args:
            directory (str, optional): _description_. Defaults to "Data/Dataset/Transcripts/".
        """
        self.directory = directory
        self.data_list = []
        self.month_to_quarter = {
            "Jan": 1, "Feb": 1, "Mar": 1,
            "Apr": 2, "May": 2, "Jun": 2, 
            "Jul": 3, "Aug": 3, "Sep": 3,
            "Oct": 4, "Nov": 4, "Dec": 4,
        }
        self.df = None
        self.train, self.test = self.load_data(singleCompany)
        self.vocab = self.make_vocab()

    def load_data(self, singleCompany=False):
        """
        Load the data from the directory and cleans it.

        Returns:
            df: returns a pandas dataframe with the following columns:
                - Year: the year of the transcript
                - Quarter: the quarter of the transcript
                - Company: the company of the transcript
                - Presentation: the prepared part of the transcript
                - QA: the Q&A part of the transcript
        """
        for folder in os.listdir(self.directory):
            if singleCompany and folder != singleCompany:
                continue
            if folder == ".DS_Store":
                continue
            for file in os.listdir(self.directory + folder):
                if file.endswith(".txt"):
                    assert (
                        file.split("-")[3].split(".")[0] == folder
                    ), "Company name does not match folder name"
                    year = file.split("-")[0]
                    month = file.split("-")[1]
                    quarter = self.month_to_quarter[month]

                    with open(self.directory + folder + "/" + file, "r") as f:
                        data = f.read()
                        pres, qa = self.clean_data(data)
                        self.data_list.append(
                            {
                                "Year": year,
                                "Quarter": quarter,
                                "Company": folder,
                                "Presentation": pres,
                                "QA": qa,
                            }
                        )

        self.df = pd.DataFrame(self.data_list, columns=["Year", "Quarter", "Company", "Presentation", "QA"])
        self.df = self.df.sort_values(by=["Company", "Year", "Quarter"]).reset_index(drop=True)
        return self.split_data()
    
    def clean_data(self, data: str):
        """
        Cleans the data by removing unwanted lines and characters.

        Args:
            data (str): The raw string data from the .txt file.

        Raises:
            ValueError: No 'Presentation' or 'OVERVIEW' found in the transcript.
            This error is to catch if the data doesn't begin with the keyword 
            'Presentation' or 'OVERVIEW'.
            ValueError: No 'Definitions' or 'Disclaimer' found in the transcript.
            This error is to catch if the data doesn't end with the keyword
            'Definitions' or 'Disclaimer'.
            ValueError: No 'Questions and Answers' found in the transcript.
            This error is to catch if the data doesn't contain the keyword
            'Questions and Answers', which is the delimiter between the
            presentation and the Q&A portions of the transcript.

        Returns:
            list, list: Returns two lists, one for the presentation and one for the Q&A.
        """
        # delete all lines before the first occurence of "Presentation" or "OVERVIEW"
        try:
            data = data.split("Presentation", 1)[1]
        except IndexError:
            try:
                data = data.split("OVERVIEW", 1)[1]
            except IndexError:
                raise ValueError("No 'Presentation' or 'OVERVIEW' found in the transcript")

        # delete all lines after the first occurence of "Definitions" or "Disclaimer"
        try:
            data = data.split("Definitions", 1)[0]
        except IndexError:
            try:
                data = data.split("Disclaimer", 1)[0]
            except IndexError:
                raise ValueError("No 'Definitions' or 'Disclaimer' found in the transcript")

        undesired_lines = [
            "Thomson Reuters", "Refinitiv", "E D I T E D",
            "Q1", "Q2", "Q3", "Q4", 
            "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", 
            "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
            "*", "=", "-",
            "Corporate Participants", "Conference Call Participiants",
            "Operator"
            ]
        
        # getting rid of names
        data = data.split("\n")
        i = 0
        while i < len(data):
            try:
                if data[i].startswith("--") and data[i+2].startswith("--"):
                    undesired_lines.append(data[i+1])
                    i += 3
                else:
                    i += 1
            except IndexError:
                break
                
        data = [line for line in data if not any(line.strip().startswith(token) for token in undesired_lines)]
        data = "\n".join(data)

        data = re.sub(r'<Sync[^>]*>', '', data) # remove <Sync> tags
        data = data.lower()
        data = data.replace("\r", " ")
        data = data.replace("\n", " ")
        data = data.replace(". ", " ")
        data = data.replace(", ", " ")
        data = data.replace("good morning", " ")
        data = data.replace("good afternoon", " ")
        data = data.replace("good evening", " ")
        data = data.replace("good day", " ")
        data = data.replace("thank you", " ")

        stops = set(stopwords.words('english'))
        table_punctuation = str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')

        data = data.translate(table_punctuation)
        if "questions and answers" in data:
            data = data.split("questions and answers", 1)
        else:
            raise ValueError("No 'Questions and Answers' found in the transcript")

        pres = [word for word in nltk.tokenize.word_tokenize(data[0]) if word not in stops]
        qa = [word for word in nltk.tokenize.word_tokenize(data[1]) if word not in stops]

        return pres, qa

    def make_vocab(self):
        """
        Creates a dictionary which has each word for each company.

        Returns:
            dict: Returns a dictionary containing a dictionary with each word.
        """
        vocab = {}
        for company in self.df["Company"].unique():
            pres_unique_words = set(self.train[self.train["Company"] == company]["Presentation"].sum())
            qa_unique_words = set(self.train[self.train["Company"] == company]["QA"].sum())
            vocab[company] = {}
            vocab[company]["Presentation"] = {w:i for i, w in enumerate(pres_unique_words)}
            vocab[company]["QA"] = {w:i for i, w in enumerate(qa_unique_words)}
        return vocab

    def split_data(self):
        """
        Split the data into training and testing sets.

        Returns:
            df, df: Returns two dataframes, one for training and one for testing.
        """
        train = self.df[self.df["Year"] != "2020"]
        test = self.df[self.df["Year"] == "2020"]
        return train, test
    
    def split_pres_qa(self, df):
        """
        Helper function which will split a given dataframe into two lists, one for the presentation
        and one for the Q&A.

        Args:
            df: The dataframe to be split.

        Returns:
            list, list: Returns two lists, one for the presentation and one for the Q&A.
        """
        pres = df["Presentation"].sum()
        qa = df["QA"].sum()
        return pres, qa
        