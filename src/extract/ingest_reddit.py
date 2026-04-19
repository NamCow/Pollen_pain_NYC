"""
Fetches r/nyc posts and comments mentioning allergy/pollen keywords.
Used as a behavioral validation signal only — not a primary model feature.

Requires Reddit API credentials. Create an app at:
  https://www.reddit.com/prefs/apps (choose "script" type)

Set these environment variables before running:
  REDDIT_CLIENT_ID
  REDDIT_CLIENT_SECRET
  REDDIT_USER_AGENT  (e.g. "pollen-pain-nyc/1.0 by <your_username>")
"""

import os
import praw
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

OUTPUT_CSV = "./data/raw/reddit_allergy_posts.csv"

SUBREDDIT = "nyc"
SEARCH_QUERY = "allergy OR pollen OR asthma OR \"itchy eyes\" OR \"hay fever\" OR \"air quality\""
START_YEAR = 2019
END_YEAR = 2024
POST_LIMIT = 1000


def get_reddit_client() -> praw.Reddit:
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = os.environ.get("REDDIT_USER_AGENT", "pollen-pain-nyc/1.0")

    if not client_id or not client_secret:
        raise EnvironmentError(
            "Missing Reddit credentials. Set environment variables:\n"
            "  REDDIT_CLIENT_ID\n"
            "  REDDIT_CLIENT_SECRET\n"
            "  REDDIT_USER_AGENT"
        )
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )


def fetch_posts(reddit: praw.Reddit) -> list[dict]:
    sub = reddit.subreddit(SUBREDDIT)
    rows = []

    for submission in sub.search(SEARCH_QUERY, sort="new", time_filter="all", limit=POST_LIMIT):
        created = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
        if not (START_YEAR <= created.year <= END_YEAR):
            continue
        rows.append({
            "id": submission.id,
            "created_utc": created.isoformat(),
            "year": created.year,
            "month": created.month,
            "title": submission.title,
            "selftext": submission.selftext,
            "score": submission.score,
            "num_comments": submission.num_comments,
            "url": submission.url,
        })

    return rows


def main():
    reddit = get_reddit_client()
    print(f"Fetching r/{SUBREDDIT} posts matching allergy keywords ({START_YEAR}-{END_YEAR})...")

    rows = fetch_posts(reddit)
    if not rows:
        print("No posts found.")
        return

    df = pd.DataFrame(rows)
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}  ({len(df):,} posts)")
    print(df.groupby("year").size().to_string())


if __name__ == "__main__":
    main()
