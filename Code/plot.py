import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, company, word_freq=None):
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

    def plot_word_freq(self, section="both", n=20):
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
            plt.figure()
            word_freq = self.word_freq[section]
            title = f"{section} word frequency for {self.company}"

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

            # plt.show()

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

    def plot_report_similarity_line(self, sim_dict, test=True, singleCompany=False):
        """
        Plots the similarity scores for one or multiple companies over the 16 reports.
        Although more informative, this plot can be difficult to read when plotting for
        multiple companies. Therefore, it is recommended to use the bar chart plot instead,
        and only use this plot when plotting for a single company.

        Args:
            sim_dict (dict): dictionary containing the similarity scores for each company.
            test (bool, optional): True if the test set is being used. Defaults to True.
            singleCompany (bool, optional): True if only one company is being plotted.
            Defaults to False.
        """
        plt.figure()
        _, ax = plt.subplots(figsize=(12, 5))
        for company, report_nums in sim_dict.items():
            quarters = []
            scores = []
            years = []
            for report_num, sim_score in report_nums.items():
                quarter = report_num % 4 + 1
                year = (report_num // 4) + (2016 if not test else 2020)

                quarters.append(quarter)
                scores.append(sim_score)
                years.append(year)

            ax.plot(scores, marker="o", linestyle="-", label=company)

        # Add main X-axis labels
        ax.set_xlim(xmin=-0.5, xmax=max(quarters) + 0.5, auto=True)
        ax.set_xticks(range(len(quarters)))
        ax.set_xticklabels([f"Q{i}" for i in quarters])

        pos = []
        pos.append(-0.5)
        for i in range(4):
            pos.append((i + 1) * 4 - 0.5)
        ax.vlines(
            pos[1:-1],
            0,
            -0.4,
            color="black",
            lw=1.5,
            clip_on=False,
            transform=ax.get_xaxis_transform(),
        )

        # Add secondary X-axis labels
        for ps0, ps1, lbl in zip(pos[:-1], pos[1:], set(years)):
            ax.text(
                (ps0 + ps1) / 2,
                -0.12,
                lbl,
                ha="center",
                weight="bold",
                size=13,
                clip_on=False,
                transform=ax.get_xaxis_transform(),
            )
        ax.set_xlabel("")
        ax.set_ylabel("Similarity Score")
        ax.legend()

        if singleCompany:
            ax.set_title(f"Similarity scores for {self.company}")
        else:
            ax.set_title("Similarity scores for all companies")

        plt.show()

    def plot_report_similarity_bar(self, sim_dict, test=True, singleCompany=False):
        """
        Plots the similarity scores for one or multiple companies over the 16 reports.
        Formats the plot as a bar chart, this is easier to read when plotting for
        multiple companies.

        Args:
            sim_dict (dict): dictionary containing the similarity scores for each company.
            test (bool, optional): True if the test set is being used. Defaults to True.
            singleCompany (bool, optional): True if only one company is being plotted.
            Defaults to False.
        """
        plt.figure()
        _, ax = plt.subplots(figsize=(12, 5))
        sorted_companies = sorted(
            sim_dict.keys(), key=lambda x: np.mean(list(sim_dict[x].values()))
        )

        for company in sorted_companies:
            report_nums = sim_dict[company]
            quarters = []
            scores = []
            years = []
            for report_num, sim_score in report_nums.items():
                quarter = report_num % 4 + 1
                year = (report_num // 4) + (2016 if not test else 2020)

                quarters.append(quarter)
                scores.append(sim_score)
                years.append(year)

            sorted_scores = sorted(scores)
            error = np.array(
                [
                    np.mean(scores) - sorted_scores[0],
                    sorted_scores[-1] - np.mean(scores),
                ]
            ).reshape(2, 1)

            ax.bar(company, np.mean(scores), yerr=error, capsize=5, label=company)

        ax.set_xticks([])
        ax.set_ylabel("Similarity Score")
        ax.legend()

        if singleCompany:
            ax.set_title(f"Similarity scores for {self.company}")
        else:
            ax.set_title("Similarity scores for all companies")

        plt.show()
