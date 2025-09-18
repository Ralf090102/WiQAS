import json
from dataclasses import dataclass
from pathlib import Path

from langchain.schema import Document


@dataclass
class BooksMetadata:
    title: str | None = None
    author: str | None = None
    translated_by: str | None = None
    year_published: str | None = None
    text: str | None = None


@dataclass
class NewsSitesMetadata:
    text: str | None = None


@dataclass
class OnlineForumsMetadata:
    author: str | None = None
    created_utc: str | None = None
    full_link: str | None = None
    text: str | None = None
    subreddit: str | None = None
    title: str | None = None
    created: float | None = None


@dataclass
class SocialMediaMetadata:
    text: str | None = None
    year: str | None = None
    month: str | None = None


@dataclass
class WikipediaMetadata:
    date: float | None = None
    title: str | None = None
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

        if "books" in data:
            metadata = BooksMetadata(
                title=data.get("title"),
                author=data.get("author"),
                translated_by=data.get("translatedBy"),
                year_published=data.get("yearPublished"),
                text=data.get("text"),
            )
        elif "news_sites" in data:
            metadata = NewsSitesMetadata(
                text=data.get("text"),
            )
        elif "online_forums" in data:
            metadata = OnlineForumsMetadata(
                author=data.get("author"),
                created_utc=data.get("created_utc"),
                full_link=data.get("full_link"),
                text=data.get("text"),
                subreddit=data.get("subreddit"),
                title=data.get("title"),
                created=data.get("created"),
            )
        elif "social_media" in data:
            metadata = SocialMediaMetadata(
                text=data.get("text"),
                year=data.get("year"),
                month=data.get("month"),
            )
        elif "wikipedia" in data:
            metadata = WikipediaMetadata(
                date=data.get("date"),
                title=data.get("title"),
                text=data.get("text"),
            )
        else:
            raise ValueError("Unknown metadata type in JSON file")

        document = Document(
            page_content=json.dumps(data),
            metadata={"subfolder_path": subfolder_path, "file_name": self.file_path.name, **metadata.__dict__},
        )

        return [document]
