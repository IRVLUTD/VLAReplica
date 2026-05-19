Task Variants Convention

Place one JSON file per task in this folder.

Recommended naming:
- task01.json
- task02.json
- task03.json
- ...
- task10.json

Each file should contain exactly 5 task instruction variants, in order.
Variant 1 corresponds to reference image 1 for that task, variant 2 to image 2, and so on.

Accepted JSON shapes:
- A plain JSON list of 5 strings
- An object with a "tasks" array containing 5 strings or 5 objects with "name"/"task" fields

Example:
[
  "Put the bread on the red plate.",
  "Put the bread on the blue plate.",
  "Put the bread on the green plate.",
  "Put the bread on the yellow plate.",
  "Put the bread on the white plate."
]
