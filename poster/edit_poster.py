"""Update pool_coach_poster_v3.pptx with the final results -> poster/pool_coach_poster_v4.pptx.

Keeps the v3 design (36 x 24 in, three columns). Text runs keep their formatting; only the words change.
Every number comes from results/ (see poster/POSTER_CONTENT.md for sources).

Usage: python poster/edit_poster.py
"""
import copy

import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

SRC, OUT = "pool_coach_poster_v3.pptx", "poster/pool_coach_poster_v4.pptx"
prs = Presentation(SRC)
slide = prs.slides[0]
tree = slide.shapes._spTree
S = {s.shape_id: s for s in slide.shapes}
next_id = [max(S) + 1]


def move(sid, x=None, y=None, w=None, h=None, dy=None):
    s = S[sid]
    if dy is not None:
        s.top = s.top + Inches(dy)
    if x is not None:
        s.left = Inches(x)
    if y is not None:
        s.top = Inches(y)
    if w is not None:
        s.width = Inches(w)
    if h is not None:
        s.height = Inches(h)


def set_text(sid, paras):
    """paras: list of strings or lists of (text, bold) runs. Reuses each paragraph's first-run format."""
    tf = S[sid].text_frame
    ps = tf.paragraphs
    while len(ps) < len(paras):
        tf._txBody.append(copy.deepcopy(ps[-1]._p))
        ps = tf.paragraphs
    for p in ps[len(paras):]:
        p._p.getparent().remove(p._p)
    for p, content in zip(tf.paragraphs, paras):
        runs = content if isinstance(content, list) else [(content, None)]
        r0 = p.runs[0]._r
        for r in p.runs[1:]:
            r._r.getparent().remove(r._r)
        prev = r0
        for k, (text, bold) in enumerate(runs):
            r = r0 if k == 0 else copy.deepcopy(r0)
            if k:
                prev.addnext(r)
            prev = r
            r.find("{http://schemas.openxmlformats.org/drawingml/2006/main}t").text = text
            rpr = r.find("{http://schemas.openxmlformats.org/drawingml/2006/main}rPr")
            if bold is not None:
                rpr.set("b", "1" if bold else "0")


def clone(sid, x, y, w, h, text):
    """Copy a text box (same style) to a new position with new text."""
    next_id[0] = max(s.shape_id for s in slide.shapes) + 1
    el = copy.deepcopy(S[sid]._element)
    el.find(".//{http://schemas.openxmlformats.org/presentationml/2006/main}cNvPr").set("id", str(next_id[0]))
    el.find(".//{http://schemas.openxmlformats.org/presentationml/2006/main}cNvPr").set("name", f"Text {next_id[0]}")
    tree.append(el)
    new = [s for s in slide.shapes if s.shape_id == next_id[0]][0]
    S[next_id[0]] = new
    next_id[0] += 1
    new.left, new.top, new.width, new.height = Inches(x), Inches(y), Inches(w), Inches(h)
    set_text(new.shape_id, [text] if isinstance(text, str) else text)
    return new.shape_id


def replace_picture(sid, path, x, y, w=None, h=None, alt=""):
    old = S[sid]
    old._element.getparent().remove(old._element)
    pic = slide.shapes.add_picture(path, Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)
    pic._element.find(".//{http://schemas.openxmlformats.org/presentationml/2006/main}cNvPr").set("descr", alt)
    return pic


# ---------------------------------------------------------------- header
set_text(3, ["Yuri Ishitoya, Ari Prezerowitz, Arjun Naik  ·  Rice University  ·  "
             "ELEC/COMP 576 Introduction to Deep Learning, Fall 2026"])

# ---------------------------------------------------------------- column 1
set_text(11, ["Beginners lose at pool without knowing why: the right shot depends on how accurately you hit it. "
              "We simulate 8-ball games with noisy players at three skill levels and train networks that predict "
              "the winner and the success of every candidate shot. They reach about 90% of a slow simulator "
              "search's success in 2 ms instead of 23 s, but skill-aware advice barely beats skill-agnostic advice."])
replace_picture(17, "figures/poster_slot_concept.png", 1.0, 11.05, h=5.45,
                alt="Same test table: best shot for a beginner (ball 7, 58%) vs. an expert (ball 2, 94%)")
move(18, x=6.25, w=5.55)

# ---------------------------------------------------------------- column 2
S[66]._element.getparent().remove(S[66]._element)  # candidate-shot picture: report only
move(67, x=12.6, y=6.10, w=10.8)
set_text(68, ["8 easiest ball-pocket pairs × 3 speeds = up to 24 candidate shots.",
              "Each simulated 16 times per skill level: success rate = shot label.",
              "Who finally wins the rack = win label.",
              "Network input: ball positions + both skills; no simulation at test time.",
              "Compared: heuristic, MLP, DeepSets, Set Transformer, CNN (3 seeds each)."])
move(68, x=12.6, y=6.70, w=10.8, h=2.35)
for sid in range(69, 80):  # skill-noise block moves up
    move(sid, dy=-3.85)
win_head = clone(69, 12.6, 13.35, 10.8, 0.55, "Who will win? A simple baseline wins")
pic = slide.shapes.add_picture("figures/poster_slot_win.png", Inches(12.6), Inches(13.95), height=Inches(3.45))
pic._element.find(".//{http://schemas.openxmlformats.org/presentationml/2006/main}cNvPr").set(
    "descr", "Win log loss: logistic regression on skills and ball counts 0.541 beats every network")
for sid in range(80, 89):  # failure cards move down a little
    move(sid, dy=0.40)

# ---------------------------------------------------------------- column 3
set_text(94, ["~90%", "of search success (heuristic: 84 to 91%)"])
set_text(96, ["2 ms", "per table, vs. 23 s for search"])

df = pd.read_csv("results/poster_shot_success.csv")  # written by src.poster_figures (shot_eval_single)
order = [("Teacher (search)", "Simulator search (teacher)", "C9C9C9"),
         ("Set Transformer", "Set Transformer, skill input", "1D5C45"),
         ("No skill input", "Set Transformer, no skill input", "8CC4AA"),
         ("3-skill head", "Set Transformer, 3-output head", "4E9C7E"),
         ("Heuristic", "Geometric heuristic", "D9731F")]
cd = CategoryChartData()
cd.categories = ["Beginner", "Intermediate", "Expert"]
for name, label, _ in order:
    r = df[df.label == label].set_index("skill").loc[["beginner", "intermediate", "expert"], "mean"]
    cd.add_series(name, [round(float(v), 3) for v in r], number_format="0.00")
chart = S[97].chart
chart.replace_data(cd)
for ser, (_, _, col) in zip(chart.plots[0].series, order):
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = RGBColor.from_string(col)
chart.chart_title.text_frame.paragraphs[0].runs[0].text = (
    "Success of the recommended shot (300 test tables, 64 fresh tries; networks: mean of 3 seeds)")
move(97, y=6.50, h=4.15)
set_text(98, ["Networks beat the heuristic for beginners and intermediates and tie it for experts. "
              "Skill input or a 3-skill head barely changes the advice."])
move(98, y=10.70, h=0.75)

dep_head = clone(69, 24.2, 11.55, 10.8, 0.55, "Does skill change the best shot? Rarely")
replace_picture(99, "figures/poster_slot_skill_dependence.png", 24.2, 12.15, w=10.8,
                alt="Best ball or pocket differs between skills on 14 to 18% of tables vs. an 11% noise floor")
set_text(100, ["Beyond the 11% noise floor, skill changes the best ball or pocket on only 3 to 7% of tables. "
               "Ignoring skill costs a beginner 2.4 points."])
move(100, y=16.15, h=0.75)
clone(100, 24.2, 16.95, 10.8, 1.0,
      [[("Also tried: ", True), ("M=32 labels (no change); ranking loss (beginner +1.1, expert -2.4 pts); "
                                  "spin (expert +1.4 pts). Set models tie within seed noise; CNN is worst.", False)]])
for sid in (101, 102, 103, 104):
    move(sid, dy=0.30)
set_text(105, ["A network gets about 90% of the search's success, 10,000x faster, but beats a geometric "
               "heuristic by at most 2 points.",
               "Skill-aware heads tie skill-agnostic ones: in our shot set, skill rarely changes the best shot.",
               "Win prediction: a logistic regression on skills + ball counts beats every network."])
move(105, y=18.90, h=2.75)

prs.save(OUT)
print("saved", OUT)
