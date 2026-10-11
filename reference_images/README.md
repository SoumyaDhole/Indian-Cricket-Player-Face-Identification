# Reference photos

Choose a small set of players you intend to support. Add one subfolder per player
and put only genuine, properly labeled face photos in that player's folder:

```text
reference_images/
  Virat_Kohli/
    kohli-match-1.jpg
    kohli-portrait.png
  Rohit_Sharma/
    sharma-portrait.jpg
```

Folder names are the displayed player labels; underscores are shown as spaces. Only
Folder names are displayed labels (underscores become spaces); the examples above
do not mean those players are preloaded. Use clear images showing exactly one
face, and obtain photos only from sources whose terms permit your use. Keep a note
of source and license for each image. The application does not download or scrape
reference photos. Unreadable or non-single-face images are skipped when the
models load; `python validate_setup.py` only checks file decoding, not face count.