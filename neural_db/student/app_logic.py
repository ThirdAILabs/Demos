import copy
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional

import requests
import thirdai

thirdai.set_seed(7)
from thirdai import licensing

from student import documents, loggers, qa, teachers
from student.models import Mach
from student.state import State

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
}


class Upvoter:
    def __init__(self, state: State, user_id: str, query: str, doc_id: int):
        self.state = state
        self.user_id = user_id
        self.query = query
        self.doc_id = doc_id

    def upvote(self):
        teachers.upvote(
            model=self.state.model,
            logger=self.state.logger,
            user_id=self.user_id,
            query=self.query,
            liked_passage_id=self.doc_id,
        )


class SearchState:
    def __init__(self, query: str, references: List[documents.Reference]):
        self._query = query
        self._references = references

    def len(self):
        return len(self._references)

    def __len__(self):
        return self.len()

    def references(self):
        return self._references


class AnswererState:
    def __init__(self, answerer: qa.QA, context_args: qa.ContextArgs):
        self._answerer = answerer
        self._context_args = context_args

    def answerer(self):
        return self._answerer

    def context_args(self):
        return self._context_args


class AppLogic:
    def __init__(self, thirdai_license: str, user_id: str, default_num_results: int):
        licensing.activate(thirdai_license)
        self._user_id = user_id
        self._savable_state: Optional[State] = None
        self._search_state: Optional[SearchState] = None
        self._answerer_state: Optional[AnswererState] = None
        self._num_search_references = default_num_results

    def from_scratch(self):
        self._savable_state = State(
            model=Mach(id_col="id", query_col="query"),
            logger=loggers.LoggerList([]),
        )
        self._search_state = None

    def in_session(self) -> bool:
        return self._savable_state is not None

    def ready_to_search(self) -> bool:
        return self.in_session() and self._savable_state.ready()

    def from_checkpoint(
        self, checkpoint_path: Path, on_progress: Callable, on_error: Callable
    ):
        try:
            self._savable_state = State.load(checkpoint_path, on_progress)
            if not isinstance(self._savable_state.logger, loggers.LoggerList):
                # TODO(Geordie / Yash): Add DBLogger to LoggerList once ready.
                self._savable_state.logger = loggers.LoggerList(
                    [self._savable_state.logger]
                )
        except Exception as e:
            self._savable_state = None
            on_error(error_msg=e.__str__())

        self._search_state = None

    def clear_session(self):
        self._savable_state = None
        self._search_state = None

    def sources(self) -> List[str]:
        return self._savable_state.documents.list_sources()

    def save(self, save_to: Path, on_progress: Callable):
        return self._savable_state.save(save_to, on_progress)

    def _file_names_by_extension(filenames):
        def is_pdf(filename: str):
            return filename.lower().endswith(".pdf")

        def is_docx(filename: str):
            return filename.lower().endswith(".docx")

        invalids = [
            name for name in filenames if not is_pdf(name) and not is_docx(name)
        ]
        pdfs = [name for name in filenames if is_pdf(name)]
        docxs = [name for name in filenames if is_docx(name)]

        return pdfs, docxs, invalids

    def _add_files_to_doclist(self, pdfs, docxs, on_error):
        doclist_copy = copy.deepcopy(self._savable_state.documents)

        document_objects = []
        try:
            if len(pdfs) > 0:
                document_objects.append(
                    documents.PDF(
                        files=pdfs,
                        expected_id_col=self._savable_state.model.get_id_col(),
                        hash_to_id_offset=self._savable_state.documents.get_source_hash_to_id_offset_map(),
                        next_id_offset=self._savable_state.documents.get_num_ids(),
                    )
                )
                self._savable_state.documents.add_document(document_objects[-1])

            if len(docxs) > 0:
                document_objects.append(
                    documents.DOCX(
                        files=docxs,
                        expected_id_col=self._savable_state.model.get_id_col(),
                        hash_to_id_offset=self._savable_state.documents.get_source_hash_to_id_offset_map(),
                        next_id_offset=self._savable_state.documents.get_num_ids(),
                    )
                )
                self._savable_state.documents.add_document(document_objects[-1])
            return document_objects
        except Exception as e:
            self._savable_state.documents = doclist_copy
            on_error(e)
            return []

    def add_files(
        self,
        file_paths: List[Path],
        on_progress: Callable,
        on_success: Callable,
        on_error: Callable,
        on_irrecoverable_error: Callable,
    ):
        if file_paths is None or len(file_paths) == 0:
            return

        filenames = [file.name for file in file_paths]
        pdfs, docxs, invalids = AppLogic._file_names_by_extension(filenames)

        if len(invalids) > 0:
            on_error(
                error_msg=f"Found files with invalid extensions {', '.join(invalids)}.\nSupported file extensions: .pdf, .docx"
            )
            return

        add_files_error_msg = [None]

        def on_add_files_error(error_msg):
            add_files_error_msg[0] = error_msg

        document_objects = self._add_files_to_doclist(
            pdfs, docxs, on_error=on_add_files_error
        )

        if add_files_error_msg[0] is not None:
            on_error(error_msg=f"Failed to add files. {add_files_error_msg}")
            return

        try:
            for i, doc in enumerate(document_objects):

                def on_current_doc_progress(fraction):
                    frac_per_doc = 1 / len(document_objects)
                    start_fraction = frac_per_doc * i
                    current_fraction = start_fraction + frac_per_doc * fraction
                    on_progress(current_fraction)

                self._savable_state.model.index_documents(
                    documents=doc,
                    on_progress=on_current_doc_progress,
                )

            self._savable_state.logger.log(
                session_id=self._user_id,
                action="Train",
                args={"files": [file.name for file in file_paths]},
            )

            on_success()

        except Exception as e:
            # If we fail during training here it's hard to guarantee that we
            # recover to a resumable state. E.g. if we're in the middle of
            # introducing new documents, we may be in a weird state where half
            # the documents are introduced while others aren't.
            # At the same time, if we fail here, then there must be something
            # wrong with the model, not how we used it, so it should be very
            # rare and probably not worth saving.
            self.clear_session()
            on_irrecoverable_error(
                error_msg=f"Failed to train model on added files. {e.__str__()}"
            )

    def add_url(
        self,
        url: str,
        scrape_depth: str,
        on_progress: Callable,
        on_success: Callable,
        on_error: Callable,
        on_irrecoverable_error: Callable,
    ):
        if not url:
            return

        try:
            response = requests.get(url, headers=HEADERS)
            if response.status_code != 200:
                on_error(
                    error_msg=f"URL is invalid with status code: {response.status_code}"
                )
                return
        except Exception as e:
            on_error(error_msg=f"URL is invalid: {e}")
            return

        scrape_depth = int(scrape_depth)
        url_doc = documents.URL(
            base_url=url,
            expected_id_col=self._savable_state.model.get_id_col(),
            hash_to_id_offset=self._savable_state.documents.get_source_hash_to_id_offset_map(),
            next_id_offset=self._savable_state.documents.get_num_ids(),
            scrape_depth=scrape_depth,
        )
        self._savable_state.documents.add_document(url_doc)

        try:
            self._savable_state.model.index_documents(
                documents=url_doc,
            )

            self._savable_state.logger.log(
                session_id="global",
                action="Train",
                args={"url": url},
            )

            on_success()

        except Exception as e:
            # If we fail during training here it's hard to guarantee that we
            # recover to a resumable state. E.g. if we're in the middle of
            # introducing new documents, we may be in a weird state where half
            # the documents are introduced while others aren't.
            # At the same time, if we fail here, then there must be something
            # wrong with the model, not how we used it, so it should be very
            # rare and probably not worth saving.
            self.clear_session()
            on_irrecoverable_error(
                error_msg=f"Failed to train model on added url. {e.__str__()}"
            )

    def clear_files(self):
        self._savable_state.documents.clear()
        self._savable_state.model.forget_documents()

    def search(self, query: str, on_error: Callable) -> List[documents.Reference]:
        try:
            result_ids = self._savable_state.model.infer_labels(
                samples=[query], n_results=self._num_search_references
            )[0]
            self._search_state = SearchState(
                query=query,
                references=[
                    self._savable_state.documents.get_reference(id) for id in result_ids
                ],
            )
            return self._search_state.references()
        except Exception as e:
            on_error(e.__str__())
            return []

    def upvote(self, idx):
        teachers.upvote(
            model=self._savable_state.model,
            logger=self._savable_state.logger,
            user_id=self._user_id,
            query=self._search_state._query,
            liked_passage_id=self._search_state.references()[idx].id(),
        )

    def associate(self, source, target, top_k_buckets):
        teachers.associate(
            model=self._savable_state.model,
            logger=self._savable_state.logger,
            user_id=self._user_id,
            text_a=source,
            text_b=target,
            top_k=top_k_buckets,
        )

    def set_answerer_state(self, answerer_state: AnswererState):
        self._answerer_state = answerer_state

    def can_answer(self) -> bool:
        return self._answerer_state is not None

    def answer(self, on_error: Callable):
        num_references = self._answerer_state.context_args().num_references
        references = self._search_state.references()[:num_references]

        # Check if "num_references" is the only context arg defined.
        # If not, we retrieve custom context for each document.
        if len(vars(self._answerer_state.context_args())) == 1:
            answers = [reference.text() for reference in references]
        else:
            answers = [
                self._savable_state.documents.get_context(
                    reference.id(), self._answerer_state.context_args()
                )
                for reference in references
            ]

        return self._answerer_state.answerer().answer(
            question=self._search_state._query,
            answers=answers,
            on_error=on_error,
        )

    def set_num_references(self, num_results: int):
        self._num_search_references = num_results

    def num_search_references(self) -> int:
        return self._num_search_references
