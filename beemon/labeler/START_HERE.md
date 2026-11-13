# 🐝 Bee labeller — start here

You get pictures from our bee videos. Your job: **put a box on every insect** in every picture.
It takes about 10–20 seconds per picture.

## 1. Start the labeller (once per session)

You need Python 3.10 or newer.

- **Linux / Mac:** open a terminal in this folder and run `bash run_labeller.sh`
- **Windows:** double-click `run_labeller.bat`

The first start installs what it needs (a few minutes). Then your browser opens the labeller.
If it does not open by itself, go to <http://localhost:8501>.

## 2. Settings (left side)

- **Your name** — type it once.
- **People labelling this set / I am person number** — if several people share the same pictures,
  everybody types the same number of people and a *different* own number (1, 2, 3 …).
  Then everybody gets different pictures.

## 3. Labelling

| to do | how |
|---|---|
| box an insect | **click on it** — a box appears |
| select a box | click on the box (it gets a white dashed line) |
| move a box | select it, then drag it |
| make a box bigger / smaller | `+` / `-` keys or the size buttons |
| delete a box | select it, then `Del` key or *Delete box* |
| type of insect | buttons at the top or keys `1`–`5`: 1 bee · 2 bumblebee · 3 hoverfly · 4 other_insect · 5 moth |
| zoom in / out | mouse wheel (bees are small — zoom in!) |
| move around | drag the picture · `0` = whole picture |
| no insect in the picture | *No insect here*, then save |
| **save and next picture** | **`Enter`** or *Save & next* |

Boxes that are already there are the **computer's guesses**. Many are wrong — dark flower centres, stems,
the orange stake. **Delete wrong boxes, add missing insects, fix the type.**

Rules:
- The box hugs the **whole insect including the wings**.
- Partly hidden but more than half visible → box it. Less than half → leave it.
- Not sure what kind of insect → **4 other_insect**. Not sure if it is an insect at all → no box.

Your work is saved on your computer after every picture; you can stop and continue later.

## 4. When you are done

Click **Download my labels** (left side) and send the downloaded `.zip` file to the person who gave you
the pictures. That's all — thank you! 🐝
