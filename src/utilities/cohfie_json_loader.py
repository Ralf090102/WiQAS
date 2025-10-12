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
    title: str | None = None            # Added title field


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
    
    def _detect_source_type(self, subfolder_path: str) -> str:
        """
        Detect the source type based on the subfolder path.
        
        Args:
            subfolder_path: The subfolder path extracted from file path
            
        Returns:
            String indicating the source type
        """
        path_lower = subfolder_path.lower()
        
        if "100year" in path_lower:
            return "100year"
        elif "bible" in path_lower:
            return "bible"
        elif "google_books" in path_lower:
            return "google_books"
        elif "gutenberg" in path_lower:
            return "gutenberg"
        elif "news_sites" in path_lower:
            return "news_sites"
        elif "wikipedia" in path_lower:
            return "wikipedia"
        else:
            return "unknown"
    
    def _extract_news_title_from_url(self, url: str) -> str:
        """
        Extract article title from URL path.
        
        Args:
            url: The URL to extract title from
            
        Returns:
            Article title extracted from URL path
        """
        if not url:
            return "News Article"
        
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            path = parsed.path
            
            # Remove leading/trailing slashes and split by slashes
            path_parts = [part for part in path.split('/') if part]
            
            if not path_parts:
                return "News Article"
            
            # Get the last part of the path (usually the article slug)
            article_slug = path_parts[-1]
            
            # Skip GMA URLs for now as they're different format
            if "gmanews.tv" in url.lower() or "gma" in url.lower():
                return "GMA News Article"
            
            # Clean up the slug to make it a proper title
            title = self._clean_url_slug_to_title(article_slug)
            
            return title if title else "News Article"
            
        except:
            return "News Article"
    
    def _clean_url_slug_to_title(self, slug: str) -> str:
        """
        Convert URL slug to proper title format.
        
        Args:
            slug: URL slug (e.g., "bukaka-photo-ni-kc-nilait-pinagpantasyahan-sa-instagram")
            
        Returns:
            Proper title format (e.g., "Bukaka Photo Ni Kc Nilait Pinagpantasyahan Sa Instagram")
        """
        if not slug:
            return ""
        
        # Replace hyphens and underscores with spaces
        title = slug.replace('-', ' ').replace('_', ' ')
        
        # Remove common file extensions
        title = title.replace('.html', '').replace('.php', '').replace('.aspx', '')
        
        # Remove trailing numbers (often article IDs)
        import re
        title = re.sub(r'\s+\d+$', '', title)
        
        # Capitalize each word
        title = ' '.join(word.capitalize() for word in title.split())
        
        return title

    def _get_title_for_source(self, item: dict, source_type: str, subfolder_path: str) -> str:
        """
        Get the appropriate title based on source type and item data.
        
        Args:
            item: The data item from JSON
            source_type: The detected source type
            subfolder_path: The subfolder path
            
        Returns:
            Appropriate title string
        """
        if source_type == "100year":
            return "Bantay-Wika"
        elif source_type == "bible":
            return "Bibliya"
        elif source_type in ["google_books", "gutenberg", "wikipedia"]:
            # These have title fields, use them if available
            return item.get("title", f"Untitled {source_type.replace('_', ' ').title()}")
        elif source_type == "news_sites":
            # Extract title from URL field based on domain
            url = item.get("url", "")
            return self._extract_news_title_from_url(url)
        else:
            # Fallback for unknown sources
            return item.get("title", "Unknown Source")

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
        
        # Detect source type from path
        source_type = self._detect_source_type(subfolder_path)

        documents = []

        # Handle top-level list structure
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    raise ValueError(f"Invalid item in JSON list: {item}")

                # Set title based on source type and existing data
                title = self._get_title_for_source(item, source_type, subfolder_path)
                
                # Determine metadata type for each item
                if all(key in item for key in ["title", "author", "translatedBy", "yearPublished", "text"]):
                    metadata = BooksMetadata(
                        title=title,
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
                        title=title,
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
                        title=title,
                        text=item.get("text"),
                    )
                elif "text" in item and source_type in ["100year", "bible"]:
                    # Handle simple text-only cases for 100year and bible sources
                    metadata = WikipediaMetadata(
                        date=item.get("date"),
                        title=title,
                        text=item.get("text"),
                    )
                else:
                    print(f"Unknown metadata type in JSON file: {self.file_path}")
                    print(f"Top-level keys: {list(item.keys())}")
                    print(f"Source type: {source_type}")
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
