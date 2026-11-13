# Help the model: collect hard negatives

**No experience needed.** About 20 minutes per video.

The model puts boxes on insects, but sometimes also on things that are **not** insects: flower centres,
buds, leaves, shadows, the orange stake. Such a mistake is called a **hard negative**. When you delete
these wrong boxes, the next model learns from your corrections and stops making the same mistakes.

## 1. Start the app (once)

Follow *Start in 3 steps* in the [README](README.md): get the code, download the model into `models/`,
then run `run.bat` (Windows) or `bash run.sh` (Linux/Mac).

## 2. Take a free video

1. Open [videos.csv](videos.csv) (or the video list in the app). Pick a video with status `todo`.
2. **Tell the team which one you take**, so nobody does it twice: write in the group chat, or edit
   `videos.csv` on GitHub (pencil icon) and put your name into `taken_by` and `taken` into `status`.
3. Download it from Sciebo: `video_records_2024/<sciebo_folder>/<file>`.
4. Put it into `videos/<last part of sciebo_folder>/`, e.g. `videos/20240724_WM2_plot18/RIMG0001.MOV`.
   (Or leave it where it is and choose it with **Browse…** in the app.)

## 3. In the app: page "Hard negatives"

1. **Your name** (top right) and **Choose a video**: click its row in the table. *On this computer*
   shows which videos the app already found; *Started by me* shows your unfinished ones.
2. **Let the model find pictures**: keep the settings and press **Find pictures**. The model looks at one
   picture every 10 seconds in the background and keeps those in which it found something (about 3 s per
   picture on a laptop). You can **start checking right away**: new pictures are added while you work.
3. **Check each picture.** Every model box has a letter (A, B, C …). On the right, **Does it move?** shows
   each box 1 second before, now and 1 second after:
   - an **insect** moves, turns or is gone in one of the three;
   - a **flower centre, bud, leaf or the stake** looks exactly the same in all three → delete that box.

   | you see a box on … | do |
   |---|---|
   | a flower centre, bud, leaf, stem, shadow, stone, the orange stake | click the box → **Delete box** (or the `Del` key) |
   | an insect (bee, bumblebee, hoverfly, fly, beetle, moth …) | keep it |
   | nothing at all is an insect in this picture | **No insect here** (removes all boxes) |
   | an insect **without** a box | click on it → a box appears (`+`/`-` change its size) |
   | you are not sure | **Skip** (the picture is left out) |

   Then **Save & next** (or `Enter`). Zoom with the mouse wheel, move the picture by dragging it, `0`
   shows the whole picture. The numbers under the picture jump to any picture (✓ = checked); *Undo this
   picture* brings back the model's boxes.
4. **Send your work**: press **Download my work (.zip)** and send the file to the person who trains the
   model (e.g. upload it to the shared Sciebo folder). You can continue later: your work is saved on your
   computer and the next download contains everything. *What you taught the model* shows all the wrong
   boxes you deleted.

When you are done, change the video's status to `done` (or tell the team).

## Already have a YOLO dataset zip from another review tool?

A zip with `images/`, `labels/` and `data.yaml` can be used as it is. If the tool also wrote a review CSV
(`review_report.csv`), send it too: it tells which video each picture came from.

## Questions

- *The app does not start*: check that Python 3.10+ is installed (`python --version`). On Windows run
  `run.bat` again after installing Python with *"Add to PATH"*.
- *"Find pictures" is slow*: that is normal on a laptop (about 2 seconds per picture). Choose 20 pictures.
- *I made a mistake on a picture*: go **Back** to it and correct it, or press
  **Undo my changes on this picture**.
