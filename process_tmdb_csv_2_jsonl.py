import pandas as pd
import json
import sys
from typing import Dict, List, Optional


def collapse_genres(j: str) -> str:
    """Convert JSON string of genres to space-separated genre names."""
    try:
        genres = []
        ar = json.loads(j)
        for a in ar:
            if isinstance(a, dict) and "name" in a:
                genres.append(a.get("name"))
        return " ".join(sorted(genres))
    except json.JSONDecodeError:
        # If not JSON, assume it's already a string of genres
        return j
    except Exception as e:
        print(f"Error processing genres: {e}")
        return ""


def combine_features(row: pd.Series, text_columns: List[str]) -> str:
    """Combine specified columns into a single text field."""
    try:
        text_parts = []
        for col in text_columns:
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                text_parts.append(str(row[col]).strip())
        return " ".join(text_parts)
    except Exception as e:
        print(f"Error combining features for row: {e}")
        return ""


def process_csv_to_vespa(
    input_file: str,
    output_file: str,
    id_column: Optional[str] = None,
    title_column: Optional[str] = None,
    text_columns: Optional[List[str]] = None,
    genre_column: Optional[str] = None
) -> None:
    """
    Process a CSV file to create a Vespa-compatible JSON format.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output JSONL file
        id_column: Name of the ID column (default: auto-detect)
        title_column: Name of the title column (default: auto-detect)
        text_columns: List of columns to combine for text search (default: auto-detect)
        genre_column: Name of the genre column (default: auto-detect)
    """
    try:
        # Read CSV and get column names
        df = pd.read_csv(input_file)
        columns = df.columns.tolist()
        print(f"Available columns: {columns}")
        
        # Auto-detect columns if not specified
        if not id_column:
            # Check for common ID column patterns
            id_candidates = ['_id', 'id', 'doc_id', 'post_id', 'hash_id']
            id_column = next((col for col in id_candidates if col in columns), None)
        
        if not title_column:
            # For Twitter data, we might use username, name, or screen_name as title
            title_candidates = ['title', 'original_title', 'name', 'username', 'screen_name', 'user.name']
            title_column = next((col for col in title_candidates if col in columns), None)
        
        if not genre_column:
            # For Twitter data, data_topic could serve as genre/category
            genre_candidates = ['genres', 'genre', 'data_topic', 'topic', 'category']
            genre_column = next((col for col in genre_candidates if col in columns), None)
        
        if not text_columns:
            # For Twitter data, prioritize content, description, and other text fields
            text_candidates = ['content', 'text', 'description', 'overview', 'location']
            text_columns = [col for col in text_candidates if col in columns]
            
            # Add genre/topic column to text if available
            if genre_column and genre_column not in text_columns:
                text_columns.append(genre_column)

        # Validate required columns
        if not id_column or id_column not in columns:
            raise ValueError(f"Could not find ID column. Available columns: {columns}")
        
        if not title_column or title_column not in columns:
            print(f"Warning: Could not find title column. Using ID as title. Available columns: {columns}")
            title_column = id_column
        
        if not text_columns:
            raise ValueError(f"No text columns identified for search. Available columns: {columns}")

        print(f"Using columns:")
        print(f"  ID: {id_column}")
        print(f"  Title: {title_column}")
        print(f"  Text: {text_columns}")
        print(f"  Genre/Topic: {genre_column}")

        # Process genres/topics if present
        if genre_column and genre_column in df.columns:
            df[f'{genre_column}_processed'] = df[genre_column].apply(lambda x: str(x) if pd.notna(x) else "")
            if genre_column in text_columns:
                text_columns[text_columns.index(genre_column)] = f'{genre_column}_processed'

        # Fill missing values
        for col in [title_column] + text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('')

        # Create combined text field
        df['text'] = df.apply(lambda x: combine_features(x, text_columns), axis=1)

        # Select and rename columns
        result_df = df[[id_column, title_column, 'text']].copy()
        result_df.rename(columns={
            id_column: 'doc_id',
            title_column: 'title'
        }, inplace=True)

        # Ensure doc_id is string format
        result_df['doc_id'] = result_df['doc_id'].astype(str)

        # Create Vespa document format
        result_df['fields'] = result_df.apply(lambda row: row.to_dict(), axis=1)
        result_df['put'] = result_df['doc_id'].apply(lambda x: f"id:hybrid-search:doc::{x}")

        # Save to JSONL
        final_df = result_df[['put', 'fields']]
        print(f"\nProcessed data preview:")
        print(final_df.head())
        final_df.to_json(output_file, orient='records', lines=True)
        print(f"\nSuccessfully processed {len(final_df)} records to {output_file}")

    except Exception as e:
        print(f"Error processing CSV: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Example usage with default column detection for Twitter data
    process_csv_to_vespa("distilled.twitter_testing_post.csv", "clean_twitter.jsonl")
    
    # Example usage with explicit column mapping for Twitter data:
    # process_csv_to_vespa(
    #     "distilled.twitter_testing_post.csv",
    #     "clean_twitter.jsonl",
    #     id_column="_id",
    #     title_column="username",
    #     text_columns=["content", "description", "location"],
    #     genre_column="data_topic"
    # )
