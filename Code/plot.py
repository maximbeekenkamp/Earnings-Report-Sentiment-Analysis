import matplotlib.pyplot as plt

from preprocessing import DataSet


class Plotter:
    def __init__(self, company, word_freq=None, sim_dict=None):
        """
        Class contains all the plotting functions for the project.

        Args:
            company (str): string containing the company name.
            word_freq (dict): dictionary containing the word frequencies for each company.
            section (str, optional): string to choose what type of data needs to be plotted. 
            Defaults to "both".
            sim_dict (dict): dictionary containing the cosine similarity between the presentation
            and QA sections for each company.
        """
        self.company = company
        if word_freq is not None:
            self.word_freq = word_freq[company]
            self.word_to_token_dict = DataSet().vocab
            self.word_to_token_dict = self.word_to_token_dict[company]
            self.token_to_word_dict = {"Presentation": {}, "QA": {}}
        
        if sim_dict is not None:
            self.sim_dict = sim_dict[company]




    def plot_word_freq(self, section="both", n=50):
        """
        Plots the n most common words for the given company and section.

        Args:
            section (str, optional): Allows toggling between displaying only the Presentation, 
            QA, or both sections. Defaults to "both".
            n (int, optional): the number of words to plot. Defaults to 50.

        Raises:
            ValueError: If the section argument is not 'Presentation', 'QA', or 'both'.

        Returns:
            None: Displays the plot.
        """
        def plot_freq(section, n):
            """
            Plots the n most common words for the given company and section.

            Args:
                section (str): the section to plot.
                n (int): the number of words to plot.
            """
            word_freq = self.word_freq[section]
            title = f"{section} word frequency for {self.company}"
            self.token_to_word_dict[section] = {
                v: k for k, v in self.word_to_token_dict[section].items()
            }
            # debugging tool:
            for i in range(len(self.word_to_token_dict[section])):
                assert (
                    self.word_to_token_dict[section][
                        self.token_to_word_dict[section][i]
                    ]
                    == i
                ), "Token to word dictionary doesn't match the word to token dictionary."
            most_common_words, most_common_occurrences = zip(*word_freq.most_common(n))

            _, ax_most_common = plt.subplots()
            ax_most_common.barh(
                y=most_common_words,
                width=most_common_occurrences,
                height=0.75,
                color="C0",
                edgecolor="black",
                zorder=100,
            )

            ax_most_common.grid(linestyle="dashed", color="#bfbfbf", zorder=-100)
            ax_most_common.set_yticks(ticks=ax_most_common.get_yticks())
            ax_most_common.set_yticklabels(labels=most_common_words, fontsize=14)
            ax_most_common.invert_yaxis()
            ax_most_common.set_title(label=title, fontsize=20)

            plt.show()

        if section == "Presentation":
            plot_freq(section, n)
        elif section == "QA":
            plot_freq(section, n)
        elif section == "both":
            plot_freq("Presentation", n)
            plot_freq("QA", n)
        else:
            raise ValueError("Argument must be 'Presentation', 'QA', or 'both'.")
        pass
