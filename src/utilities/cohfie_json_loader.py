import json
from dataclasses import dataclass
from pathlib import Path

from langchain.schema import Document


@dataclass
class BooksMetadata:
    title: str | None = None            #
    author: str | None = None           #
    translated_by: str | None = None
    year_published: str | None = None   #
    text: str | None = None


@dataclass
class NewsSitesMetadata:
    text: str | None = None
    category: str | None = None
    date: str | None = None             #
    tags: list[str] | None = None       
    url: str | None = None              #


# @dataclass
# class OnlineForumsMetadata:
#     author: str | None = None
#     created_utc: str | None = None
#     full_link: str | None = None
#     text: str | None = None
#     subreddit: str | None = None
#     title: str | None = None
#     created: float | None = None


# @dataclass
# class SocialMediaMetadata:
#     text: str | None = None
#     year: str | None = None
#     month: str | None = None


@dataclass
class WikipediaMetadata:
    date: float | None = None           #
    title: str | None = None            #
    text: str | None = None


class CohfieJsonLoader:
    """
    Loader for COHFIE JSON files. This class reads JSON files and extracts their content.
    Additionally, it saves the subfolder names as metadata in the Document object.
    """

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def load(self) -> list[Document]:
        """
        Load the JSON file and return a list of Document objects.

        Returns:
            list[Document]: List of Document objects with content and metadata.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        if self.file_path.suffix.lower() != ".json":
            raise ValueError(f"Unsupported file type: {self.file_path.suffix}")

        with open(self.file_path, encoding="utf-8") as f:
            data = json.load(f)

        # Extract subfolder names as metadata
        subfolder_path = ".".join(self.file_path.parts[-7:-1])

        documents = []

        # Handle top-level list structure
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    raise ValueError(f"Invalid item in JSON list: {item}")

                # Determine metadata type for each item
                if all(key in item for key in ["title", "author", "translatedBy", "yearPublished", "text"]):
                    metadata = BooksMetadata(
                        title=item.get("title"),
                        author=item.get("author"),
                        translated_by=item.get("translatedBy"),
                        year_published=item.get("yearPublished"),
                        text=item.get("text"),
                    )
                elif all(key in item for key in ["text", "category", "date", "tags", "url"]):
                    metadata = NewsSitesMetadata(
                        text=item.get("text") or "",
                        category=item.get("category") or "",
                        date=item.get("date") or 0,
                        tags=", ".join(item.get("tags") or []),
                        url=item.get("url") or "",
                    )
                # elif all(
                #     key in item for key in ["author", "created_utc", "full_link", "text", "subreddit", "title", "created"]
                # ):
                #     metadata = OnlineForumsMetadata(
                #         author=item.get("author"),
                #         created_utc=item.get("created_utc"),
                #         full_link=item.get("full_link"),
                #         text=item.get("text"),
                #         subreddit=item.get("subreddit"),
                #         title=item.get("title"),
                #         created=item.get("created"),
                #     )
                # elif all(key in item for key in ["text", "year", "month"]):
                #     metadata = SocialMediaMetadata(
                #         text=item.get("text"),
                #         year=item.get("year"),
                #         month=item.get("month"),
                #     )
                elif all(key in item for key in ["date", "title", "text"]):
                    metadata = WikipediaMetadata(
                        date=item.get("date"),
                        title=item.get("title"),
                        text=item.get("text"),
                    )
                else:
                    print(f"Unknown metadata type in JSON file: {self.file_path}")
                    print(f"Top-level keys: {list(item.keys())}")
                    raise ValueError("Unknown metadata type in JSON file")

                text_content = item.get("text", "")

                # Remove 'text' from metadata dict to avoid duplication
                meta_dict = {k: v for k, v in metadata.__dict__.items() if k != "text"}

                document = Document(
                    page_content=text_content,
                    metadata={"subfolder_path": subfolder_path, "file_name": self.file_path.name, **meta_dict},
                )
                documents.append(document)

        else:
            raise ValueError(f"Unsupported JSON structure in file: {self.file_path}")

        return documents
