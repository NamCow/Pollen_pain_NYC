# MongoDB Collections — Pollen & Pain

MongoDB stores unstructured / semi-structured source data that doesn't fit the relational schema.

## reddit_posts

Raw Reddit posts crawled from r/nyc, r/asthma, r/allergies, etc.

```json
{
  "_id": "<reddit post id>",
  "subreddit": "nyc",
  "title": "...",
  "selftext": "...",
  "score": 42,
  "num_comments": 7,
  "created_utc": 1680000000,
  "created_date": "2023-03-28",
  "year_month": "2023-03",
  "url": "https://reddit.com/r/nyc/...",
  "keywords": ["pollen", "asthma", "allergy"]
}
```

## pollen_raw

Daily pollen readings as returned by the AAAAI / IQVIA API before normalization.

```json
{
  "_id": "<date>",
  "date": "2023-04-15",
  "source": "aaaai",
  "TREE": 320,
  "GRASS": 12,
  "WEED": 5,
  "MOLD": 0,
  "raw_payload": { ... }
}
```

## air_quality_raw

Raw AirNow / EPA API responses before aggregation.

```json
{
  "_id": "<date>-<parameter>",
  "date": "2023-04-15",
  "parameter": "PM2.5",
  "aqi": 42,
  "category": "Good",
  "site_name": "Queens College",
  "raw_payload": { ... }
}
```
