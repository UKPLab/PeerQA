from typing import Iterator, Literal, Union

import pandas as pd
from tqdm.auto import tqdm


class PaperLoader:
    def __init__(self, papers_file: str):
        self.df = pd.read_json(papers_file, lines=True)

    def __call__(
        self,
        granularity: Literal["sentences", "paragraphs"],
        template: str = None,
        show_progress: bool = True,
    ) -> Iterator[tuple[str, list[str], list[str]]]:
        """Yields a the paper_id, sentence/paragraph index and sentence/paragraph."""

        for paper_id, paper_df in tqdm(
            self.df.groupby(["paper_id"]),
            total=self.df.paper_id.nunique(),
            disable=not show_progress,
            ncols=80,
        ):
            paper_id = paper_id[0]

            # iterate over each paper
            paper_df = paper_df.sort_values("idx")

            if template is not None:

                assert "{content}" in template, "Template must have \{content\}."

                # Get title and abstract from paper
                if "{title}" in template:
                    title = self.get_title(paper_id)
                else:
                    title = None
                if "{abstract}" in template:
                    abstract = paper_df[
                        (paper_df.last_heading.str.lower() == "abstract")
                    ].content.values.tolist()
                    assert len(abstract), f"No abstract content found for {paper_id=}."
                    abstract = " ".join(abstract)
                else:
                    abstract = None
            else:
                # set default template to content
                title, abstract = None, None
                template = "{content}"

            document_ids = []
            documents = []
            if granularity == "sentences":
                # collect all sentences in the paper
                for _, sentence in paper_df.iterrows():
                    document_ids.append(f"{sentence.pidx}/{sentence.sidx}")
                    documents.append(
                        template.format(
                            title=title, abstract=abstract, content=sentence.content
                        )
                    )
            elif granularity == "paragraphs":
                # collect all paragraphs in the paper
                paragraphs_df = paper_df.groupby("pidx").agg(
                    {"content": lambda x: " ".join(x)}
                )
                for idx, paragraph in paragraphs_df.iterrows():
                    document_ids.append(idx)
                    documents.append(
                        template.format(
                            title=title, abstract=abstract, content=paragraph.content
                        )
                    )
            else:
                raise ValueError(granularity)
            yield paper_id, document_ids, documents

    def get_title(self, paper_id: str) -> str:
        title = self.df[
            (self.df.paper_id == paper_id) & (self.df.type == "title")
        ].content.values
        assert len(title) == 1, f"Multiple titles found for {paper_id=}."
        title = title[0]
        return title


class QuestionLoader:
    def __init__(self, qa_file: str):
        self.df = pd.read_json(qa_file, lines=True)
        self.df["has_annoated_answer_evidence"] = self.df.answer_evidence_mapped.apply(
            self._has_annotated_answer_evidence
        )

    @staticmethod
    def _has_annotated_answer_evidence(answer_evidence: Union[None, list[dict]]):
        if answer_evidence is None:
            return False
        return any(ae["idx"] != [None] for ae in answer_evidence)

    def questions_by_paper_id_with_answer_evidence(
        self, paper_id: str, include_unanswerable: bool = False
    ) -> tuple[list[str], list[str]]:
        cond_paper_id = self.df.paper_id == paper_id
        cond_has_annotated_answer_evidence = self.df.has_annoated_answer_evidence
        cond_unanswerable = self.df.answerable.notna()
        if include_unanswerable:
            cond = cond_paper_id & (
                cond_has_annotated_answer_evidence | cond_unanswerable
            )
        else:
            cond = cond_paper_id & cond_has_annotated_answer_evidence

        return self.df[cond][["question_id", "question"]].values.T.tolist()

    def questions_with_answer_evidence(
        self, progress: bool = True
    ) -> Iterator[tuple[str, list[str], list[str]]]:
        paper_ids = self.df.paper_id.unique()
        for paper_id in tqdm(paper_ids, disable=not progress, ncols=80):
            yield paper_id, *self.questions_by_paper_id_with_answer_evidence(
                paper_id=paper_id
            )

    def questions_un_answerable(self, paper_id, include_un_answerable: bool = True):
        if include_un_answerable:
            # include unanswerable questions,
            # i.e. any question with answerable True or False, but not None
            m = self.df.answerable_mapped.notna()
        else:
            # only include answerable questions
            m = self.df.answerable_mapped == True
        return self.df[(self.df.paper_id == paper_id) & m][
            ["question_id", "question", "answerable_mapped"]
        ].values.T.tolist()


class HydeLoader:
    def __init__(self, hyde_file: str) -> None:
        self.df = pd.read_json(hyde_file, lines=True)

    def passages_by_question_id(self, question_id: str) -> list[str]:
        results = self.df[self.df.question_id == question_id].result
        assert len(results) == 1
        results = results.iloc[0]
        passages = [c["message"]["content"] for c in results["choices"]]
        return passages
