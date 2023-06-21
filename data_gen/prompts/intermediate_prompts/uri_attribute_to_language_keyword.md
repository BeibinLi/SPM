You are a helper bot to index a database.
A user types a human-readable question into the search box and the URI to the search is: {uri}
We know she/he is looking for an attribute: {attribute}.
This attribute could take value(s) in {{values}} (there may be unmentioned values).
Can you guess human-readable questions that the user use to search?

Use theory of mind to guess what the user is looking for.
First extract keywords from the URI and attribute, then generate the question.

Use this format for a group of answer:

KEYWORDS: keywords go here
QUESTION: user question goes here

Generate as many groups as you can, and for each group, make the keyword as concise as possible.
